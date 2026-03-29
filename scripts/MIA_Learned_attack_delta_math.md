# Learned Attack 使用 Delta（Differential）的详细数学过程（结合代码）

## 1. 数据来源与记号

- 每个 **(target_idx, round)** 对应一条「目标行」在某一轮下的 member 与 control 两套合成数据。
- **pair_metrics.csv**：对每个 (target_idx, round) 从 **member 侧**（该 target 在训练集里时生成的 in/out）算出一组数值特征，列名带后缀 `_m`。
- **pair_metrics_control.csv**：同一 (target_idx, round) 从 **control 侧**（该 target 不在训练集里、用 control 数据生成的 in/out）算出同样维度的特征，列名带后缀 `_c`。

代码（`_compute_learned_aucs` / `_get_learned_scores_df`）：

```python
merged = df_m.merge(df_c, on=id_cols, how="inner", suffixes=("_m", "_c"))  # 391, 502
# merged 的每一行 = 一个 (target_idx, round)，含 feat_m 与 feat_c
feat_m = [f"{c}_m" for c in feature_cols ...]   # 398, 508
feat_c = [f"{c}_c" for c in feature_cols ...]
M = merged[feat_m].replace(...).fillna(0).to_numpy()   # 410, 514  → shape (N, d)
C = merged[feat_c].replace(...).fillna(0).to_numpy()   # 411, 515  → shape (N, d)
```

记：
- \( N \) = 配对数 = `len(merged)`；
- \( d \) = 特征维数（如 `FEATURE_CANDIDATES` 的个数）；
- \( \mathbf{M}_i \in \mathbb{R}^d \) = 第 \( i \) 对的 **member 特征**（第 \( i \) 行 `M[i]`）；
- \( \mathbf{C}_i \in \mathbb{R}^d \) = 第 \( i \) 对的 **control 特征**（第 \( i \) 行 `C[i]`）。

---

## 2. Delta（差分）向量的定义

对每一对 \( i \) 定义 **差分向量**：

\[
\boldsymbol{\delta}_i = \mathbf{M}_i - \mathbf{C}_i \in \mathbb{R}^d.
\]

含义：同一 (target, round) 下，「member 侧特征」减去「control 侧特征」，得到该配对上的**差分信号**。

代码（两处一致）：

```python
delta = M - C   # 419, 521  → shape (N, d)，第 i 行 = δ_i
```

---

## 3. Naive Learned：不用 delta，直接对 M/C 分类

- **样本构造**：每个配对 \( i \) 贡献 **两个**样本  
  - 输入 \( \mathbf{M}_i \)，标签 1（member）；  
  - 输入 \( \mathbf{C}_i \)，标签 0（control）。  
- **特征矩阵与标签**（2N 行）：

\[
X_{\text{naive}} = \begin{bmatrix} \mathbf{M}_1 \\ \vdots \\ \mathbf{M}_N \\ \mathbf{C}_1 \\ \vdots \\ \mathbf{C}_N \end{bmatrix} \in \mathbb{R}^{2N \times d}, \qquad
y_{\text{naive}} = (\underbrace{1,\ldots,1}_{N},\underbrace{0,\ldots,0}_{N})^\top.
\]

代码：

```python
# 413-416, 516-518
X_naive = np.vstack([M, C])           # [M; C]，前 N 行 M，后 N 行 C
y_naive = np.concatenate([np.ones(N), np.zeros(N)])
groups_naive = np.concatenate([np.arange(N), np.arange(N)])  # 同一对的两行同组，CV 时同进 train 或 test
```

- **分类器**：在 \( X_{\text{naive}}, y_{\text{naive}} \) 上做带 GroupKFold 的 CV，训练 \( f_{\text{naive}}: \mathbb{R}^d \to [0,1] \)（如 LR 的 P(member)）。
- **分数**：  
  - member 分数 = \( f_{\text{naive}}(\mathbf{M}_i) \)；  
  - control 分数 = \( f_{\text{naive}}(\mathbf{C}_i) \)。

---

## 4. Delta（Differential）Learned：用 δ 与 -δ 做输入

- **样本构造**：每个配对 \( i \) 仍贡献 **两个**样本，但输入改为 **差分**：  
  - 输入 \( \boldsymbol{\delta}_i = \mathbf{M}_i - \mathbf{C}_i \)，标签 1（member）；  
  - 输入 \( -\boldsymbol{\delta}_i \)，标签 0（control）。  
- **特征矩阵与标签**（2N 行）：

\[
X_{\text{delta}} = \begin{bmatrix} \boldsymbol{\delta}_1 \\ \vdots \\ \boldsymbol{\delta}_N \\ -\boldsymbol{\delta}_1 \\ \vdots \\ -\boldsymbol{\delta}_N \end{bmatrix} = \begin{bmatrix} \mathbf{M}_1 - \mathbf{C}_1 \\ \vdots \\ \mathbf{M}_N - \mathbf{C}_N \\ \mathbf{C}_1 - \mathbf{M}_1 \\ \vdots \\ \mathbf{C}_N - \mathbf{M}_N \end{bmatrix}, \qquad
y_{\text{delta}} = (\underbrace{1,\ldots,1}_{N},\underbrace{0,\ldots,0}_{N})^\top.
\]

代码：

```python
# 418-422, 520-522
delta = M - C
X_delta = np.vstack([delta, -delta])   # 前 N 行 = +δ，后 N 行 = -δ
y_delta = np.concatenate([np.ones(N), np.zeros(N)])
groups_delta = np.repeat(np.arange(N), 2)   # 第 i 对的两行 (δ_i, -δ_i) 同组
```

- **分类器**：在 \( X_{\text{delta}}, y_{\text{delta}} \) 上做同样的 GroupKFold CV，训练 \( f_{\text{delta}}: \mathbb{R}^d \to [0,1] \)。
- **分数**：  
  - member 分数 = \( f_{\text{delta}}(\boldsymbol{\delta}_i) \)；  
  - control 分数 = \( f_{\text{delta}}(-\boldsymbol{\delta}_i) \)。

代码（out-of-fold 概率即横坐标）：

```python
# 558-561
pred_delta = cross_val_predict(pipe, X_delta, y_delta, cv=splits_delta, method="predict_proba")[:, 1]
out["delta_learned_member"] = pred_delta[:N]    # f_delta(δ_i) for i=1..N
out["delta_learned_control"] = pred_delta[N:]   # f_delta(-δ_i) for i=1..N
```

---

## 5. 为何 Delta 更易区分：数学直觉

- **Naive**：分类器看到的是「一整段特征」\( \mathbf{M}_i \) 或 \( \mathbf{C}_i \)。Member 与 control 的特征分布可能整体偏移不大，只是局部有差异，导致两类在特征空间里重叠多，分离难。
- **Delta**：分类器只看 **差分** \( \boldsymbol{\delta}_i \) 和 \( -\boldsymbol{\delta}_i \)。  
  - Member 侧：\( \boldsymbol{\delta}_i = \mathbf{M}_i - \mathbf{C}_i \) 刻画「有 target 在训练集时相对 control 多出来的部分」。  
  - Control 侧：\( -\boldsymbol{\delta}_i \) 正好是反向，天然以 0 为对称。  
  因此学习目标是「区分 +δ 与 -δ」，等价于在差分空间里区分方向，往往比在原始 M/C 空间里区分两类更容易，AUC 更高。

---

## 6. 与 k-NN / Density 里 Delta 的对应关系

在 k-NN 与 Density 中（`_worker_one_pair` 等）：
- 先对 member 和 control 各自算一个**标量**分数 \( s_m, s_c \)（如 \( \bar{d}_{\text{out}} - \bar{d}_{\text{in}} \)）；
- Delta 标量 = \( \delta = s_m - s_c \)，member 得分取 \( +\delta \)，control 得分取 \( -\delta \)。

在 Learned 中：
- Member/control 是 **向量** \( \mathbf{M}_i, \mathbf{C}_i \)；
- Delta **向量** \( \boldsymbol{\delta}_i = \mathbf{M}_i - \mathbf{C}_i \)；
- Member 输入 \( \boldsymbol{\delta}_i \)，control 输入 \( -\boldsymbol{\delta}_i \)，再交给 LR 得到标量分数（概率）。

因此：**Learned 的 Delta 做法 = 在特征空间里先做「member − control」的差分，再用分类器在差分向量上区分 member（+δ）与 control（−δ）**；与 k-NN/Density 的「标量差分、±δ 作为分数」是同一思想在向量特征上的推广。

---

## 7. 小结（公式 + 代码位置）

| 步骤 | 数学 | 代码（约行） |
|------|------|--------------|
| 1. 配对特征 | \( \mathbf{M}_i, \mathbf{C}_i \) 来自 pair_metrics / pair_metrics_control | 391, 410-411, 502, 514-515 |
| 2. 差分向量 | \( \boldsymbol{\delta}_i = \mathbf{M}_i - \mathbf{C}_i \) | 419, 521 `delta = M - C` |
| 3. Delta 输入矩阵 | \( X_{\text{delta}} = [\boldsymbol{\delta}_1;\ldots;\boldsymbol{\delta}_N; -\boldsymbol{\delta}_1;\ldots;-\boldsymbol{\delta}_N] \) | 420 `X_delta = np.vstack([delta, -delta])` |
| 4. 标签 | \( y_{\text{delta}} = (1,\ldots,1,0,\ldots,0)^\top \) | 421 `y_delta = np.concatenate([np.ones(N), np.zeros(N)])` |
| 5. 训练与预测 | GroupKFold CV，LR 得到 \( f_{\text{delta}}(\boldsymbol{\delta}_i) \)、\( f_{\text{delta}}(-\boldsymbol{\delta}_i) \) | 456-458, 558-561 `cross_val_predict(..., method="predict_proba")[:, 1]` |
| 6. 输出分数 | member 分数 = \( f_{\text{delta}}(\boldsymbol{\delta}_i) \)，control 分数 = \( f_{\text{delta}}(-\boldsymbol{\delta}_i) \) | 510-511 `delta_learned_member` / `delta_learned_control` |

以上即 Learned attack 使用 delta（differential）的完整数学过程与代码对应关系。
