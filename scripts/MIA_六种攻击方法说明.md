# 六种 MIA 攻击方法详解（结合 shadow_mia_from_raw.py）

## 一、整体情景与数据

### 1.1 场景设定

- **目标**：做 **成员推断攻击（Membership Inference Attack, MIA）**——判断一条「目标行」（target）是否曾出现在生成模型的**训练集**里。
- **影子模型**：对每个候选 target，我们跑两套实验：
  - **Member（成员）**：该 target **在**训练集里时，用模型生成 **in-set** 与 **out-set** 合成数据（in 与 target 同分布倾向，out 与 target 不同分布倾向）。
  - **Control（对照）**：该 target **不在**训练集里时（用另一份 control 训练数据），同样生成 **control_in** 与 **control_out** 合成数据。
- **配对**：按 **(target_idx, round)** 配对。同一 target、同一 round 下，有 4 张表：
  - `in_df`：member 的 in-set 合成样本
  - `out_df`：member 的 out-set 合成样本
  - `c_in_df`：control 的 in-set 合成样本
  - `c_out_df`：control 的 out-set 合成样本

代码里通过 `_collect_paired_paths` 收集所有满足「同时存在 member in/out 与 control in/out」的 (target_idx, round_k)，每个这样的「对」会进 `_worker_one_pair` 算一次分数。

### 1.2 两类「攻击思路」

- **Naive（朴素）**：**不显式用 control 做减法**。对 member 侧算一个分数，对 control 侧算一个分数，然后把「member 的分数」当正类（label=1）、「control 的分数」当负类（label=0），用所有 (score, label) 算一个 AUC。  
  含义：单看这条 target 在「自己的 in/out 对比」下有多像 member，与「另一条 target 在 control 的 in/out 对比」下有多像 member，用分数区分两类。

- **Delta（差分）**：**显式用 member − control 做减法**。对同一 (target, round) 先算 member 分数 \(s_m\) 和 control 分数 \(s_c\)，令 \(\delta = s_m - s_c\)。  
  约定：**member 行**得分 = \(\delta\)（label=1），**control 行**得分 = \(-\delta\)（label=0）。  
  含义：用「同一 target、同一 round 下，member 比 control 强多少」作为信号；减号是约定好的（member 用 +δ，control 用 −δ），不是随机。

下面 6 种方法 = 2(Naive / Delta) × 3(k-NN / Density / Learned)。

---

## 二、方法 1：Naive k-NN（k=1, 8, 32）

### 2.1 思路

只看「目标行」到 **in-set** 与 **out-set** 的**距离**：若到 in 更近、到 out 更远，则更像 member（训练时见过 target）。  
**不**用 control 数据参与「算分公式」，只分别对 member 和 control 各算一个分数，再汇总成 (score, label) 算 AUC。

### 2.2 代码与公式

**Step 1：距离与 k-NN 平均距离**

- 数值列：标准化后欧氏距离（按列数归一化）；类别列：0/1 不一致比例；混合距离 = 数值距离 + 类别距离。  
  见 `_mixed_type_distances`（约 92–123 行）。
- 对 **member**：  
  - `d_in` = target 到 `in_df` 每行距离，`d_out` = target 到 `out_df` 每行距离。  
  - \(\bar{d}_{\text{in}}^{(m)}(k)\) = 到 in 的最近 k 个点距离的平均，\(\bar{d}_{\text{out}}^{(m)}(k)\) = 到 out 的最近 k 个点距离的平均。  
  见 `_compute_scores_one_row`（约 152–186 行），内部用 `_avg_dist_k`（约 125–131 行）。

**Step 2：Naive k-NN 分数（标量）**

```python
# 约 189–193 行
def _naive_score_knn(avg_in: float, avg_out: float) -> float:
    """Naive k-NN：用 in/out 对比，越大越像 member。d_out - d_in（out 远、in 近则正）。"""
    return float(avg_out - avg_in)
```

即：
\[
s = \bar{d}_{\text{out}} - \bar{d}_{\text{in}}.
\]
同一 (target, round) 下：
- **Member 侧**：用 member 的 in/out 算 \(s_m = \bar{d}_{\text{out}}^{(m)} - \bar{d}_{\text{in}}^{(m)}\)；
- **Control 侧**：用 control 的 in/out 算 \(s_c = \bar{d}_{\text{out}}^{(c)} - \bar{d}_{\text{in}}^{(c)}\)。

**Step 3：汇总成 AUC**

在 `_process_one`（约 509–538 行）中：
- 对每个 (target, round)，若 \(s_m\) 有限则加入 `(s_m, 1)`，若 \(s_c\) 有限则加入 `(s_c, 0)`。
- 最后对所有 (score, label) 用 `roc_auc_score` 算 AUC，并取 \(\max(\text{AUC}, 1-\text{AUC})\)（约 555–561 行）。

### 2.3 情景小结

- **输入**：每个 (target, round) 的 member in/out、control in/out 四张表；target 行。
- **输出**：每个 pair 两个分数 \(s_m, s_c\)；全 run 一个 `auc_naive_knn_k1`（以及 k8、k32）。
- **含义**：用「到 out 远、到 in 近」的程度（标量）区分「这条是 member 算的」vs「这条是 control 算的」。

---

## 三、方法 2：Naive Density

### 3.1 思路

用 **k-NN 密度**（DOMIAS 式）：在 target 处，in-set 的密度高、out-set 的密度低，则更像 member。  
同样**不**在公式里减 control，只对 member 和 control 各算一个密度分数，再 (score, label) 算 AUC。

### 3.2 代码与公式

**Step 1：log 密度**

`_knn_density_at_target`（约 134–148 行）：
\[
\text{log\_density} \propto -d_{\text{eff}} \cdot \ln(r_k),\quad r_k = \text{第 } k \text{ 近邻距离}.
\]
对 member 的 in/out 各算 \(\ell_{\text{in}}^{(m)},\ \ell_{\text{out}}^{(m)}\)；对 control 的 in/out 各算 \(\ell_{\text{in}}^{(c)},\ \ell_{\text{out}}^{(c)}\)（\(k\) 用 `DENSITY_K=8`，\(d_{\text{eff}}\) 为特征维数）。

**Step 2：Naive density 分数**

```python
# 约 196–200 行
def _naive_score_density(log_density_in: float, log_density_out: float) -> float:
    """Naive density：log(density_in) - log(density_out)，member 在 in 集密度更高则正。"""
    return float(log_density_in - log_density_out)
```

即 \(d = \ell_{\text{in}} - \ell_{\text{out}}\)。  
Member 侧：\(d_m\)；Control 侧：\(d_c\)。

**Step 3：汇总**

同 Naive k-NN：每个 pair 贡献 \((d_m, 1)\) 和 \((d_c, 0)\)，得到 `auc_naive_density`。

### 3.3 情景小结

- **输入**：同上，四张表 + target。
- **输出**：每个 pair 的 \(d_m, d_c\)；全 run 一个 `auc_naive_density`。
- **含义**：用「in 处密度高、out 处密度低」的程度区分 member 与 control 两条曲线。

---

## 四、方法 3：Naive Learned（LR）

### 4.1 思路

用 **pair_metrics** 里预计算好的多种统计量（MMD、距离、列统计等）作为特征，训练一个**分类器**（逻辑回归）区分「这一行是 member 的统计」还是「control 的统计」。  
**不**做 member−control 相减，每条行就是一个特征向量 + 一个标签（member=1 / control=0）。

### 4.2 数据来源

- `pair_metrics.csv`：每个 (target_idx, round) 一行，对应 **member** 的 in/out 算出的指标（如 mmd_numeric, min_dist_in, min_dist_out, delta_min_dist, num_*_diff, cat_tv_* 等）。
- `pair_metrics_control.csv`：同一 (target_idx, round) 下 **control** 的 in/out 算出的同类指标。  
若不存在则先跑 `run_shadow_metrics_pipeline`（约 348–369 行 `_ensure_pair_metrics`）。

### 4.3 代码与公式

**Step 1：特征与标签**

在 `_compute_learned_aucs`（约 371–482 行）中：
- 将两表按 (target_idx, round) **inner merge**，得到每对一行，列带后缀 `_m`（member）和 `_c`（control）。
- 特征列取 `FEATURE_CANDIDATES` 在两张表里都存在的数值列（约 79–89 行），inf 填 0。
- **Naive**：  
  \(X_{\text{naive}} = [M; C]\)（前 N 行是 member 的 11 维特征，后 N 行是 control 的 11 维特征），  
  \(y_{\text{naive}} = [1,\ldots,1, 0,\ldots,0]\)（member=1, control=0）。  
  见约 415–418 行。

**Step 2：按对划分、避免泄漏**

- 为每行赋予 **group**：同一 (target_idx, round) 的 member 行和 control 行同一 group。
- 用**按组划分**的 CV（显式 splits 或 GroupKFold），保证同一对不会一半在 train、一半在 test（约 426–444 行）。
- 标准化用 **Pipeline(StandardScaler(), LogisticRegression())**，只在 train 上 fit（约 473 行）。

**Step 3：AUC**

- 5-fold 按组 CV，每折用 `scoring="roc_auc"`，最后对 test AUC 取平均，并 \(\max(\text{auc}, 1-\text{auc})\)（约 474–476 行），得到 `auc_naive_learned_lr`。

### 4.4 情景小结

- **输入**：pair_metrics.csv、pair_metrics_control.csv（每对一行，多列统计量）。
- **输出**：一个标量 `auc_naive_learned_lr`。
- **含义**：用「单行特征向量」区分 member 行与 control 行，不显式做减法。

---

## 五、方法 4：Delta k-NN（k=1, 8, 32）

### 5.1 思路

在**同一 (target, round)** 上，用 member 的 k-NN 分数减去 control 的 k-NN 分数，得到标量 \(\delta\)；再约定 member 得分 = \(\delta\)、control 得分 = \(-\delta\)，用这些 (score, label) 算 AUC。  
即：用「member 比 control 多出来的 in/out 对比强度」做信号。

### 5.2 代码与公式

在 `_worker_one_pair`（约 332–339 行）中，与 Naive 一样先算 \(s_m, s_c\)，然后：

```python
delta_knn[k] = (sig_m - sig_c) if np.isfinite(sig_m) and np.isfinite(sig_c) else float("nan")
```

即 \(\delta = s_m - s_c\)。

在 `_process_one`（约 526–533 行）中，每个 pair 贡献：
- \((\delta, 1)\)（member），  
- \((-\delta, 0)\)（control）。  

再对所有 (score, label) 算 AUC，得到 `auc_delta_knn_k1`（及 k8、k32）。

### 5.3 情景小结

- **输入**：同 Naive k-NN（四张表 + target）。
- **输出**：每个 pair 一个 \(\delta\)，再展开为两个 (score, label)；全 run 一个 `auc_delta_knn_k*`。
- **含义**：用「同一 target、同一 round 下，member 与 control 的 k-NN 分数差」及其负值区分两类。

---

## 六、方法 5：Delta Density

### 6.1 思路

与 Delta k-NN 完全平行：\(\delta_d = d_m - d_c\)，member 得分 = \(\delta_d\)，control 得分 = \(-\delta_d\)。

### 6.2 代码与公式

约 337–338 行：

```python
delta_density = (d_m - d_c) if np.isfinite(d_m) and np.isfinite(d_c) else float("nan")
```

约 534–538 行：每个 pair 贡献 \((\delta_d, 1)\) 和 \((-\delta_d, 0)\)，得到 `auc_delta_density`。

### 6.3 情景小结

- **输入**：同 Naive Density。
- **输出**：`auc_delta_density`。
- **含义**：用「member 与 control 的密度分数差」做差分信号。

---

## 七、方法 6：Delta Learned（LR）

### 7.1 思路

用 **member 特征向量 − control 特征向量** 作为「差分特征」：同一 (target, round) 得到 \(\delta_{\text{vec}} = M - C\)（11 维）。  
每个 pair 贡献两个样本：\((\delta_{\text{vec}}, 1)\) 与 \((-\delta_{\text{vec}}, 0)\)，再训练 LR（+ 按组 CV）得到 AUC。

### 7.2 代码与公式

在 `_compute_learned_aucs` 中（约 420–424 行）：

```python
delta = M - C
X_delta = np.vstack([delta, -delta])
y_delta = np.concatenate([np.ones(N), np.zeros(N)])
groups_delta = np.repeat(np.arange(N), 2)
```

即：每对一行 \(M\)、一行 \(C\)，相减得 \(\delta\)；\(X_{\text{delta}}\) 为 \([δ_1,\ldots,δ_N, -\delta_1,\ldots,-\delta_N]\)，\(y\) 为 \([1,…,1,0,…,0]\)。  
同样用按组 CV + Pipeline(StandardScaler(), LR)，得到 `auc_delta_learned_lr`（约 477–480 行）。

### 7.3 情景小结

- **输入**：pair_metrics / pair_metrics_control（与 Naive Learned 相同）。
- **输出**：`auc_delta_learned_lr`。
- **含义**：用「member 与 control 的特征差向量」及其负向量训练分类器，等价于在「差分空间」里做 MIA。

---

## 八、汇总表（6 组 × 代码位置与公式）

| 组 | 方法 | 分数/特征定义 | 汇总为 AUC 的方式 | 主要代码位置 |
|----|------|----------------|-------------------|--------------|
| 1 | Naive k-NN (k=1,8,32) | \(s = \bar{d}_{\text{out}} - \bar{d}_{\text{in}}\)，member/control 各算 \(s_m, s_c\) | \((s_m,1), (s_c,0)\) → roc_auc | `_naive_score_knn`, `_process_one` |
| 2 | Naive Density | \(d = \ell_{\text{in}} - \ell_{\text{out}}\)，各算 \(d_m, d_c\) | \((d_m,1), (d_c,0)\) → roc_auc | `_naive_score_density`, `_process_one` |
| 3 | Naive Learned | 行特征 \(M\) / \(C\)，不相减 | \(X=[M;C], y=[1;0]\)，按组 CV + LR → mean test AUC | `_compute_learned_aucs` X_naive, groups_naive |
| 4 | Delta k-NN (k=1,8,32) | \(\delta = s_m - s_c\) | \((\delta,1), (-\delta,0)\) → roc_auc | `delta_knn[k]`, `_process_one` |
| 5 | Delta Density | \(\delta_d = d_m - d_c\) | \((\delta_d,1), (-\delta_d,0)\) → roc_auc | `delta_density`, `_process_one` |
| 6 | Delta Learned | \(\delta_{\text{vec}} = M - C\) | \(X=[\delta;-\delta], y=[1;0]\)，按组 CV + LR → mean test AUC | `_compute_learned_aucs` X_delta, groups_delta |

k-NN 与 Density 的原始量（距离、密度）在 `_worker_one_pair` 里用 `_compute_scores_one_row` 对 **member 的 in/out** 和 **control 的 in/out** 各算一遍；Learned 的原始特征来自 `shadow_pair_metrics` 的 pipeline 产出，在 `_compute_learned_aucs` 中读入并做 Naive/Delta 两种构造与 CV。

---

## 九、数据流简图

```
run_dir (一次实验)
├── shadow/target_*/synthetic_round_*_in.csv, *_out.csv     [member in/out]
├── shadow/target_*/synthetic_round_*_control_in.csv, *_control_out.csv [control in/out]
└── shadow_pair_metrics/
    ├── pair_metrics.csv         [member 每对一行，多列统计]
    └── pair_metrics_control.csv [control 每对一行，同列]

每个 (target_idx, round) 一对 →
  Naive k-NN / Density: 用 in/out 算 s_m, s_c 或 d_m, d_c → 收集 (s_m,1),(s_c,0) 或 (δ,1),(-δ,0)
  Learned: 读 pair_metrics 两表 → merge → Naive: [M;C]+y 或 Delta: [δ;-δ]+y → 按组 CV + LR → AUC
```

以上即六种攻击方法在代码中的完整对应与情景说明。
