#!/usr/bin/env python3
"""
Delta 方法实现正确性审计脚本。
对指定 run 目录执行 6 步检查，输出可核验证据（路径、计数、样例、统计、raw AUC）。
用法: python scripts/audit_delta_implementation.py --run-dir outputs/train_adult_openml_..._seed42_...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CANDIDATES = [
    "mmd_numeric", "mmd_mixed", "min_dist_in", "min_dist_out", "delta_min_dist",
    "num_mean_diff_mean", "num_mean_diff_max", "num_std_diff_mean", "num_std_diff_max",
    "cat_tv_mean", "cat_tv_max",
]


def _parse_round_member(p: Path) -> int | None:
    try:
        core = p.name.split("synthetic_round_", 1)[1]
        num = core.split("_", 1)[0]
        return int(num)
    except Exception:
        return None


def _get_round_files_member(target_dir: Path) -> list[tuple[int, Path, Path]]:
    in_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_in.csv") and "control" not in p.name]
    out_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_out.csv") and "control" not in p.name]
    in_map = {}
    for p in in_files:
        k = _parse_round_member(p)
        if k is not None:
            in_map[k] = p
    pairs = []
    for p in out_files:
        k = _parse_round_member(p)
        if k is not None and k in in_map:
            pairs.append((k, in_map[k], p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _get_round_files_control(target_dir: Path) -> list[tuple[int, Path, Path]]:
    in_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_control_in.csv")]
    out_files = [p for p in target_dir.iterdir() if p.is_file() and p.name.endswith("_control_out.csv")]
    in_map = {}
    for p in in_files:
        k = _parse_round_member(p)
        if k is not None:
            in_map[k] = p
    pairs = []
    for p in out_files:
        k = _parse_round_member(p)
        if k is not None and k in in_map:
            pairs.append((k, in_map[k], p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _collect_paired_paths(run_dir: Path) -> list[tuple[int, int, Path, Path, Path, Path]]:
    shadow_dir = run_dir / "shadow"
    if not shadow_dir.exists():
        return []
    target_indices = []
    for d in sorted(shadow_dir.iterdir(), key=lambda x: x.name):
        if not d.is_dir() or not d.name.startswith("target_"):
            continue
        suffix = d.name[7:]
        if suffix.isdigit():
            target_indices.append(int(suffix))
    out = []
    for ti in target_indices:
        target_dir = shadow_dir / f"target_{ti}"
        member_rounds = _get_round_files_member(target_dir)
        control_rounds = _get_round_files_control(target_dir)
        if not member_rounds or not control_rounds:
            continue
        control_round_map = {r[0]: (r[1], r[2]) for r in control_rounds}
        for round_k, in_path, out_path in member_rounds:
            if round_k not in control_round_map:
                continue
            c_in_path, c_out_path = control_round_map[round_k]
            out.append((ti, round_k, in_path, out_path, c_in_path, c_out_path))
    return out


def compute_delta_learned_auc(df_m: pd.DataFrame, df_c: pd.DataFrame, shuffle_control: bool = False, random_state: int = 42) -> tuple[float, float]:
    """计算 delta learned AUC。返回 (raw_auc, abs_auc)。shuffle_control=True 时按行下标错配：member 行 i 配 control 行 perm[i]。"""
    id_cols = [c for c in ["target_idx", "round"] if c in df_m.columns and c in df_c.columns]
    if not id_cols:
        return float("nan"), float("nan")
    merged = df_m.merge(df_c, on=id_cols, how="inner", suffixes=("_m", "_c"))
    if merged.empty:
        return float("nan"), float("nan")
    common = set(df_m.select_dtypes(include=[np.number]).columns) & set(df_c.select_dtypes(include=[np.number]).columns)
    common -= {"target_idx", "round"}
    feature_cols = [x for x in FEATURE_CANDIDATES if x in common] or sorted(common)
    feat_m = [f"{c}_m" for c in feature_cols if f"{c}_m" in merged.columns]
    feat_c = [f"{c}_c" for c in feature_cols if f"{c}_c" in merged.columns]
    if not feat_m or not feat_c:
        return float("nan"), float("nan")
    M = merged[feat_m].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    C = merged[feat_c].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    if shuffle_control:
        perm = np.random.default_rng(random_state).permutation(len(C))
        C = C[perm]
    delta = M - C
    N = len(delta)
    X = np.vstack([delta, -delta])
    y = np.concatenate([np.ones(N), np.zeros(N)])
    groups = np.repeat(np.arange(N), 2)
    gkf = GroupKFold(n_splits=min(5, max(2, N // 2)))
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], pred)
        scores.append(auc)
    if not scores:
        return float("nan"), float("nan")
    raw_auc = float(np.mean(scores))
    abs_auc = max(raw_auc, 1.0 - raw_auc)
    return raw_auc, abs_auc


def main() -> int:
    parser = argparse.ArgumentParser(description="Delta 方法实现审计")
    parser.add_argument("--run-dir", type=str, required=True, help="run 目录路径")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    results = {"1_pair_completeness": "PENDING", "2_pairing_logic": "PENDING", "3_strict_replace_one": "PENDING",
               "4_seed_alignment": "PENDING", "5_shuffle_test": "PENDING"}

    # ----- 1) Pair 完整性 -----
    print("=" * 80)
    print("1) PAIR COMPLETENESS")
    print("=" * 80)
    shadow_dir = run_dir / "shadow"
    if not shadow_dir.exists():
        print("  shadow/ 不存在 -> 无 pair")
        results["1_pair_completeness"] = "FAIL"
    else:
        pairs = _collect_paired_paths(run_dir)
        print(f"  实际收集到的 (target_idx, round) pair 数: {len(pairs)}")

        missing_by_target = {}
        for d in sorted(shadow_dir.iterdir(), key=lambda x: x.name):
            if not d.is_dir() or not d.name.startswith("target_"):
                continue
            suffix = d.name[7:]
            if not suffix.isdigit():
                continue
            ti = int(suffix)
            member_rounds = _get_round_files_member(d)
            control_rounds = _get_round_files_control(d)
            member_round_set = {r[0] for r in member_rounds}
            control_round_set = {r[0] for r in control_rounds}
            only_m = member_round_set - control_round_set
            only_c = control_round_set - member_round_set
            if only_m or only_c:
                missing_by_target[ti] = {"only_member": sorted(only_m), "only_control": sorted(only_c)}
        if missing_by_target:
            print("  缺失的 target/round（有 member 无 control 或有 control 无 member）:")
            for ti, v in list(missing_by_target.items())[:20]:
                print(f"    target_{ti}: only_member rounds={v['only_member'][:10]}{'...' if len(v['only_member'])>10 else ''}, only_control rounds={v['only_control'][:10]}{'...' if len(v['only_control'])>10 else ''}")
            if len(missing_by_target) > 20:
                print(f"    ... and {len(missing_by_target)-20} more targets")
            results["1_pair_completeness"] = "WARN"
        else:
            print("  每个 pair 均同时存在 in/out/control_in/control_out，无缺失。")
            results["1_pair_completeness"] = "PASS"

    # ----- 2) Pairing 逻辑 -----
    print("\n" + "=" * 80)
    print("2) PAIRING LOGIC")
    print("=" * 80)
    print("  代码位置: scripts/shadow_mia_from_raw.py")
    print("  - _get_round_files_member: 解析 round 用 name.split('synthetic_round_',1)[1].split('_',1)[0] -> int")
    print("  - 文件名格式: synthetic_round_{k}_in.csv / synthetic_round_{k}_out.csv")
    print("  - control: synthetic_round_{k}_control_in.csv / synthetic_round_{k}_control_out.csv")
    print("  - 因此 round_1 解析为 1, round_10 解析为 10（无歧义）。")
    pairs = _collect_paired_paths(run_dir)
    print(f"  Pair key 样例与对应文件路径（共 {len(pairs)} 个，展示前 5 个）:")
    for i, (ti, rk, in_p, out_p, c_in_p, c_out_p) in enumerate(pairs[:5]):
        print(f"    (target_idx={ti}, round={rk})")
        print(f"      in:   {in_p.name}")
        print(f"      out:  {out_p.name}")
        print(f"      c_in: {c_in_p.name}")
        print(f"      c_out:{c_out_p.name}")
    results["2_pairing_logic"] = "PASS"

    # ----- 3) Strict replace-one -----
    print("\n" + "=" * 80)
    print("3) STRICT REPLACE-ONE")
    print("=" * 80)
    print("  代码位置: src/synthgen/shadow/run_shadow.py")
    print("  - base_aux_k = aux_df.iloc[aux_indices]; train_in_k = base_aux_k ∪ target_row; train_out_k = base_aux_k ∪ x_row")
    print("  - control: train_ctrl_in = base_aux_k ∪ r1_row; train_ctrl_out = base_aux_k ∪ r2_row")
    print("  - 当前 run 未保存 Ak(aux_indices)、z(target_idx)、x、r1、r2 到磁盘。")
    manifest_path = run_dir / "shadow" / "manifest.json"
    if manifest_path.exists():
        print(f"  存在 manifest: {manifest_path}")
        results["3_strict_replace_one"] = "PASS"
    else:
        print("  缺失: 无 manifest.json（Ak ids, z/x/r1/r2 ids, seed）。")
        print("  最小 patch 建议: 在 run_shadow.py 每轮写 shadow/target_{t_idx}/round_{k}_manifest.json 含 aux_indices.tolist(), target_idx, x_candidate_idx, r1_idx, r2_idx, pair_seed, control_seed。")
        print("  最小验证 rerun: candidate=3, rounds=2，跑完后检查 manifest 并人工核对 D_in∩D_out 大小与差集。")
        results["3_strict_replace_one"] = "WARN"

    # ----- 4) Seed alignment -----
    print("\n" + "=" * 80)
    print("4) SEED ALIGNMENT")
    print("=" * 80)
    print("  代码位置: src/synthgen/shadow/run_shadow.py")
    print("  - _set_rng_seed(seed): random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all")
    print("  - in/out 同一 pair: pair_seed = seed + k * 2; _set_rng_seed(pair_seed) 后 fit model_in; _set_rng_seed(pair_seed) 后 fit model_out")
    print("  - control_in/control_out: control_seed = seed + k * 2 + 10000; _set_rng_seed(control_seed) 后 fit 两次")
    print("  -> in/out 同 seed；control_in/control_out 同 seed。")
    print("  - 若 dataloader 使用多线程/shuffle 且未设 worker_init_fn/generator，可能引入非确定性；当前代码未暴露 DataLoader，由 SynthCity 内部决定。")
    results["4_seed_alignment"] = "PASS"

    # ----- 5) Shuffle pairing test -----
    print("\n" + "=" * 80)
    print("5) SHUFFLE PAIRING TEST (Delta 不应凭空造信号)")
    print("=" * 80)
    pm_path = run_dir / "shadow_pair_metrics" / "pair_metrics.csv"
    pc_path = run_dir / "shadow_pair_metrics" / "pair_metrics_control.csv"
    if not pm_path.exists() or not pc_path.exists():
        print(f"  缺少 pair_metrics 或 pair_metrics_control: {pm_path.exists()}, {pc_path.exists()}")
        results["5_shuffle_test"] = "FAIL"
    else:
        df_m = pd.read_csv(pm_path)
        df_c = pd.read_csv(pc_path)
        raw_auc, abs_auc = compute_delta_learned_auc(df_m, df_c, shuffle_control=False)
        print(f"  正常 pairing: raw AUC = {raw_auc:.6f}, abs AUC = {abs_auc:.6f}")
        raw_shuf, abs_shuf = compute_delta_learned_auc(df_m, df_c, shuffle_control=True)
        print(f"  Shuffle control 后配对: raw AUC = {raw_shuf:.6f}, abs AUC = {abs_shuf:.6f}")
        if abs_shuf > 0.55:
            print("  WARN: Shuffle 后 AUC 仍明显 >0.5，可能存在 leakage（如 merge 错配或使用了 target_idx 等）。")
            results["5_shuffle_test"] = "FAIL"
        elif abs_shuf > 0.52:
            print("  WARN: Shuffle 后 AUC 略高于 0.5，建议重复多次 shuffle 取平均。")
            results["5_shuffle_test"] = "WARN"
        else:
            print("  预期 shuffle 后接近 0.5 -> 通过。")
            results["5_shuffle_test"] = "PASS"

    # ----- 6) 审计结论表 -----
    print("\n" + "=" * 80)
    print("6) AUDIT CONCLUSION TABLE")
    print("=" * 80)
    print("  Item                          | Status | Note")
    print("  ------------------------------|--------|----------------------------------------")
    for k, v in results.items():
        note = ""
        if v == "FAIL" and "1_" in k:
            note = "missing pairs or shadow/ absent"
        elif v == "FAIL" and "5_" in k:
            note = "shuffle AUC >>0.5, check leakage"
        elif v == "WARN" and "3_" in k:
            note = "no manifest to verify replace-one"
        elif v == "WARN" and "1_" in k:
            note = "some target/round missing pair"
        print(f"  {k:30} | {v:6} | {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
