#!/usr/bin/env python
"""
检查 Shadow 复用模式下，每个 target 是否真的拥有严格的 leave-one-out (in,out) pair。

用法：
    python scripts/check_shadow_l1o.py /path/to/run_dir

示例：
    python scripts/check_shadow_l1o.py \
        /data_storage/lixiao/research_proj_xiao/jcl/TabRisk/outputs/train_adult_openml_preprocess_monotonic_train1000_synth1000_ctgan_300iter_20260209_053225

要求：
    - 该 run 目录已经完成 shadow 生成；
    - 且是 reuse_mode=true 的运行（会有 reuse_datasets.csv / reuse_usage.csv）。
"""

import sys
from pathlib import Path

import pandas as pd


def check_shadow_reuse_run(run_dir: str) -> None:
    run_dir = Path(run_dir).resolve()
    shadow_dir = run_dir / "shadow"

    manifest_path = shadow_dir / "target_manifest.csv"
    reuse_datasets_path = shadow_dir / "reuse_datasets.csv"
    reuse_usage_path = shadow_dir / "reuse_usage.csv"

    if not manifest_path.exists():
        print(f"[ERROR] {manifest_path} 不存在，可能还没跑 shadow。")
        return
    if not reuse_datasets_path.exists() or not reuse_usage_path.exists():
        print(
            f"[ERROR] 未找到 {reuse_datasets_path.name} / {reuse_usage_path.name}，"
            f"本脚本当前只支持 reuse_mode=true 的实验。"
        )
        return

    manifest = pd.read_csv(manifest_path)
    reuse_datasets = pd.read_csv(reuse_datasets_path)
    reuse_usage = pd.read_csv(reuse_usage_path)

    # 为了快速查找，构建 dataset_id -> 行 的字典
    ds_by_id = {int(row["dataset_id"]): row for _, row in reuse_datasets.iterrows()}

    all_ok = True

    for _, mrow in manifest.iterrows():
        t_idx = int(mrow["target_idx"])
        target_cand_idx = int(mrow["candidate_row_idx"])

        usage_t = reuse_usage[reuse_usage["target_idx"] == t_idx]
        if usage_t.empty:
            print(f"[WARN] target_{t_idx}: 在 reuse_usage.csv 中没有记录，可能没有任何 round。")
            continue

        # 按 round 分组
        for round_k, grp in usage_t.groupby("round"):
            roles = set(grp["role"])
            if roles != {"in", "out"}:
                print(f"[ERROR] target_{t_idx} round_{round_k}: 角色集合 = {roles}，不是 {{'in','out'}}。")
                all_ok = False
                continue

            in_row = grp[grp["role"] == "in"].iloc[0]
            out_row = grp[grp["role"] == "out"].iloc[0]

            in_ds = ds_by_id[int(in_row["dataset_id"])]
            out_ds = ds_by_id[int(out_row["dataset_id"])]

            # 1) in/out 必须使用相同 base_id（意味着同一份 base_aux）
            if int(in_ds["base_id"]) != int(out_ds["base_id"]):
                print(
                    f"[ERROR] target_{t_idx} round_{round_k}: "
                    f"in.base_id={in_ds['base_id']} != out.base_id={out_ds['base_id']}"
                )
                all_ok = False

            # 2) in 的 candidate_row_idx 必须等于该 target 的 candidate_row_idx
            if int(in_ds["candidate_row_idx"]) != target_cand_idx:
                print(
                    f"[ERROR] target_{t_idx} round_{round_k}: "
                    f"in.candidate_row_idx={in_ds['candidate_row_idx']} "
                    f"!= target_candidate_row_idx={target_cand_idx}"
                )
                all_ok = False

            # 3) out 的 candidate_row_idx 必须 != target_cand_idx
            if int(out_ds["candidate_row_idx"]) == target_cand_idx:
                print(
                    f"[ERROR] target_{t_idx} round_{round_k}: "
                    f"out.candidate_row_idx 与 target 相同 = {target_cand_idx}"
                )
                all_ok = False

            # 4) （可选检查）out 的 is_member 应为 0（non-member）
            if int(out_ds["is_member"]) != 0:
                print(
                    f"[WARN] target_{t_idx} round_{round_k}: out.is_member={out_ds['is_member']}，"
                    f"不是 non-member（在极端回退逻辑下可能发生，请根据需要判断是否接受）。"
                )

    if all_ok:
        print(
            f"[OK] 所有 target 的 (in,out) 对在元数据上均满足严格 leave-one-out "
            f"（同一 base_id，in 使用 target 行，out 使用非 target 行）。"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scripts/check_shadow_l1o.py /path/to/run_dir")
        sys.exit(1)
    check_shadow_reuse_run(sys.argv[1])

