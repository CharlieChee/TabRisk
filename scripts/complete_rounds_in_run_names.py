#!/usr/bin/env python3
"""
检查 outputs 下 shadow run 目录名是否包含 rounds 信息；
若缺少则从 run_dir/train_config.yaml（或 shadow 相关配置）读取 num_shadow_rounds，
并可选地将目录重命名为补齐 rounds 后的名称。

可执行权限（可选）：
  chmod +x scripts/complete_rounds_in_run_names.py
用法：
  # 仅检查并打印，不重命名
  python scripts/complete_rounds_in_run_names.py
  python scripts/complete_rounds_in_run_names.py --outputs-dir outputs

  # 实际重命名
  python scripts/complete_rounds_in_run_names.py --rename
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def has_rounds_in_name(dir_name: str) -> bool:
    """目录名中是否已包含 roundsXX（如 rounds10, rounds20）。"""
    return bool(re.search(r"_rounds\d+_", f"_{dir_name}_"))


def get_rounds_from_config(run_dir: Path) -> int | None:
    """从 run_dir/train_config.yaml 读取 shadow.num_shadow_rounds。"""
    cfg_path = run_dir / "train_config.yaml"
    if not cfg_path.is_file():
        return None
    if OmegaConf is None:
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
        v = OmegaConf.select(cfg, "shadow.num_shadow_rounds", default=None)
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def candidate_pattern(dir_name: str) -> str | None:
    """匹配 shadow 段中的 candidate 部分，用于插入 rounds。如 _candidate100_。"""
    m = re.search(r"(_bs\d+)(_candidate\d+)", dir_name)
    if m:
        return m.group(2)  # _candidate100
    return None


def build_new_name(dir_name: str, rounds: int) -> str:
    """在目录名中在 candidate 前插入 rounds{N}。保持 ..._bs256_candidate100_... 顺序。"""
    # 在 _candidate 前插入 _rounds{N}
    return re.sub(r"(_bs\d+)(_candidate\d+)", r"\1_rounds" + str(rounds) + r"\2", dir_name, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查并可选补齐 outputs 下 run 目录名中的 rounds 信息")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="outputs 目录路径（默认 outputs）",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="是否执行重命名；默认仅打印",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="与 --rename 一起使用时只打印将要执行的重命名，不实际执行",
    )
    args = parser.parse_args()
    outputs_dir: Path = args.outputs_dir
    if not outputs_dir.is_dir():
        print(f"目录不存在: {outputs_dir}")
        return

    # 只处理像 shadow run 的目录：含 candidate 且为目录
    pattern = re.compile(r"candidate\d+", re.IGNORECASE)
    run_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and pattern.search(d.name)]

    missing_rounds: list[tuple[Path, str, int]] = []  # (run_dir, new_name, rounds)
    has_rounds: list[Path] = []
    no_config_or_rounds: list[Path] = []

    for run_dir in sorted(run_dirs):
        name = run_dir.name
        if has_rounds_in_name(name):
            has_rounds.append(run_dir)
            continue
        rounds = get_rounds_from_config(run_dir)
        if rounds is None:
            no_config_or_rounds.append(run_dir)
            continue
        if not candidate_pattern(name):
            no_config_or_rounds.append(run_dir)
            continue
        new_name = build_new_name(name, rounds)
        if new_name == name:
            continue
        missing_rounds.append((run_dir, new_name, rounds))

    print("===== 已包含 rounds 的目录（跳过）=====")
    for d in has_rounds[:20]:
        print(f"  {d.name}")
    if len(has_rounds) > 20:
        print(f"  ... 共 {len(has_rounds)} 个")
    print()

    print("===== 无 train_config 或无法解析 rounds（跳过）=====")
    for d in no_config_or_rounds[:20]:
        print(f"  {d.name}")
    if len(no_config_or_rounds) > 20:
        print(f"  ... 共 {len(no_config_or_rounds)} 个")
    print()

    print("===== 缺少 rounds、已从配置读取并可补齐 =====")
    if not missing_rounds:
        print("  （无）")
        return
    for run_dir, new_name, rounds in missing_rounds:
        print(f"  rounds={rounds}")
        print(f"    旧: {run_dir.name}")
        print(f"    新: {new_name}")
        if args.rename and not args.dry_run:
            new_path = run_dir.parent / new_name
            if new_path.exists():
                print(f"    跳过：目标已存在 {new_path}")
            else:
                run_dir.rename(new_path)
                print(f"    已重命名 -> {new_path.name}")
    if args.rename and args.dry_run:
        print("\n（未实际重命名，因使用了 --dry-run）")
    elif not args.rename and missing_rounds:
        print("\n如需实际重命名，请加上 --rename；可先加 --rename --dry-run 预览。")


if __name__ == "__main__":
    main()
