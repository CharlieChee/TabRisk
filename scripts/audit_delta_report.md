# Delta 方法实现正确性审计报告

**Run:** `outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_234247/`

**执行方式（可复现）：**
```bash
cd /path/to/TabRisk
python scripts/audit_delta_implementation.py --run-dir "outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_1000iter_bs256_rounds20_candidate100_control_seed42_20260220_234247"
```

---

## 1) Pair 完整性

**检查内容：** 统计实际收集到的 (target_idx, round) pair 数；对每个 pair 必须同时存在 in / out / control_in / control_out；列出缺失的 target/round。

**证据来源：**
- 路径：`run_dir/shadow/`，每个 `target_{i}/` 下文件 `synthetic_round_{k}_in.csv`, `*_out.csv`, `*_control_in.csv`, `*_control_out.csv`。
- 脚本逻辑：`_collect_paired_paths(run_dir)` 仅当 member 与 control 均存在该 round 时才计入；审计脚本额外遍历每个 target 的 member_rounds 与 control_rounds，差集即缺失。

**可核验：**
- 输出 “实际收集到的 (target_idx, round) pair 数: N”。
- 若有缺失，输出 “缺失的 target/round” 列表（only_member / only_control rounds）。

---

## 2) Pairing 逻辑是否正确

**检查内容：** 定位收集 pair 的函数及 key 解析方式；确认不会错配（如 round_1 vs round_10）。

**代码位置：** `scripts/shadow_mia_from_raw.py`
- `_get_round_files_member`（约 48–77 行）：解析 `synthetic_round_{k}_in.csv` / `*_out.csv`，`k = int(name.split("synthetic_round_",1)[1].split("_",1)[0])`。
- `_get_round_files_control`（shadow_pair_metrics.py 121–148 行）：同上格式，`*_control_in.csv` / `*_control_out.csv`。
- `_collect_paired_paths`（256–279 行）：对每个 target_i 取 member_rounds ∩ control_rounds（按 round_k 匹配），得到 (ti, round_k, in_path, out_path, c_in_path, c_out_path)。

**结论：** round 由数字段解析，`synthetic_round_1_*` → 1，`synthetic_round_10_*` → 10，无歧义。

**可核验：** 脚本输出至少 5 个 pair 的 (target_idx, round) 及对应 4 个文件名，可人工核对 round 与文件名一致。

---

## 3) Strict replace-one 是否成立

**检查内容：** 训练集构造是否满足 D_in ∩ D_out = N-1 且仅差 {z} vs {x}；control 两训练集仅差 1 条；四训练集共享同一 Ak。

**代码位置：** `src/synthgen/shadow/run_shadow.py`
- 约 177–189 行（及 503–517 行）：`rng = np.random.default_rng(seed + t_idx*10000 + k)`；`aux_indices = rng.choice(...)`；`base_aux_k = aux_df.iloc[aux_indices]`；`train_in_k = base_aux_k ∪ target_row`；`train_out_k = base_aux_k ∪ x_row`（x 为 non-member 且 ≠ target）。
- 约 212–223 行（539–565 行）：control 为 `base_aux_k ∪ r1_row` 与 `base_aux_k ∪ r2_row`，r1≠r2 且均≠target。

**当前 run 缺失：** 未将 Ak（aux_indices）、target_idx、x、r1、r2、seed 写入磁盘，故无法从现有 run 反推验证交集大小。

**最小 patch 建议：** 在每轮写完后增加写入 `shadow/target_{t_idx}/round_{k}_manifest.json`，内容示例：
```json
{"aux_indices": [...], "target_idx": 0, "x_candidate_idx": 42, "r1_idx": 1, "r2_idx": 2, "pair_seed": 84, "control_seed": 10084}
```
**最小验证 rerun：** `candidate=3, rounds=2`，跑完后用 manifest 与 candidate/aux 表人工核对 D_in ∩ D_out 与差集。

**审计结论：** 代码逻辑满足 replace-one；当前 run 无 manifest → WARN，需补 manifest 方可做可复现验证。

---

## 4) Seed alignment 是否真实生效

**检查内容：** 训练入口是否统一设置 random/numpy/torch/cuda seed；in/out 同 seed；control_in/control_out 同 seed。

**代码位置：** `src/synthgen/shadow/run_shadow.py`
- `_set_rng_seed(seed)`（74–84 行）：`random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`。
- in/out（195–210 行）：`pair_seed = seed + k * 2`；`_set_rng_seed(pair_seed)` 后 fit model_in；再次 `_set_rng_seed(pair_seed)` 后 fit model_out。
- control（227–235 行）：`control_seed = seed + k * 2 + 10000`；两次 `_set_rng_seed(control_seed)` 分别 fit model_ctrl_in / model_ctrl_out。

**结论：** in/out 使用同一 pair_seed；control_in/control_out 使用同一 control_seed；RNG 在每次 fit 前重置。

**潜在不确定性：** SynthCity 内部若使用 DataLoader 且多线程/ shuffle 未固定，可能引入非确定性；当前代码未暴露 DataLoader，未做 worker_init_fn 等设置。

---

## 5) Delta 不应“凭空造信号”：Shuffle pairing test

**检查内容：** 正常配对算 delta_learned AUC（raw + abs）；将 control 行随机打乱再配对，重算 AUC，预期应接近 0.5。

**实现：** `scripts/audit_delta_implementation.py` 中 `compute_delta_learned_auc(..., shuffle_control=True)`：
- 先按 (target_idx, round) 做 inner merge 得到 N 行；
- 取 member 特征 M 与 control 特征 C；
- shuffle 时对 C 按行做 `perm = np.random.default_rng(rs).permutation(N)`，`C = C[perm]`，即 member 行 i 与 control 行 perm[i] 配对（打破真实配对）；
- 再用 delta = M - C 训练 GroupKFold LR，得到 AUC。

**可核验：** 脚本输出 “正常 pairing: raw AUC = x, abs AUC = y” 与 “Shuffle control 后配对: raw AUC = x', abs AUC = y'”。若 shuffle 后 abs AUC 仍明显 >0.5，可能存在 leakage（如特征中含 target_idx/round 等）。

**说明：** 同时输出 raw AUC，不仅 max(AUC, 1-AUC)。

---

## 6) 审计结论表（脚本输出）

| Item                          | Status | Note |
|-------------------------------|--------|------|
| 1_pair_completeness           | PASS/WARN/FAIL | 缺失 pair 或 shadow/ 缺失 |
| 2_pairing_logic               | PASS   | round 解析无歧义 |
| 3_strict_replace_one          | WARN   | 无 manifest，无法从磁盘验证 |
| 4_seed_alignment              | PASS   | 代码中 in/out 与 control 同 seed |
| 5_shuffle_test                | PASS/WARN/FAIL | shuffle 后 AUC≈0.5 则通过 |

**FAIL 时建议：**
- 1_pair_completeness FAIL：补全缺失 round 的 control 或 member 文件，或检查 shadow 生成逻辑。
- 5_shuffle_test FAIL：检查 pair_metrics 是否含 target_idx/round 等作为特征；检查 merge 是否按 (target_idx, round) 正确配对且 shuffle 后确已错配。
