import argparse
import os
import re
from typing import Dict, List, Optional, Tuple


CHECKPOINT_PATTERN = re.compile(r"epoch(\d+)\.pth$")


def _collect_checkpoints(directory: str) -> List[Tuple[int, str]]:
    checkpoints: List[Tuple[int, str]] = []
    for filename in os.listdir(directory):
        match = CHECKPOINT_PATTERN.fullmatch(filename)
        if not match:
            continue
        checkpoints.append((int(match.group(1)), filename))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _normalize_segmented_cfg(segmented_cfg: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if segmented_cfg is None:
        return None
    if not isinstance(segmented_cfg, dict):
        raise TypeError("segmented_cfg must be a dict or None.")

    required_keys = ("epoch_seg", "interval_after_seg", "interval_before_seg")
    missing_keys = [key for key in required_keys if key not in segmented_cfg]
    if missing_keys:
        raise ValueError(
            "segmented_cfg requires keys: "
            "epoch_seg, interval_after_seg, interval_before_seg. "
            f"Missing keys: {missing_keys}"
        )
    return {key: int(segmented_cfg[key]) for key in required_keys}


def _should_keep_epoch_legacy(epoch_num: int, latest_epoch_num: int, k: int) -> bool:
    return epoch_num == latest_epoch_num or epoch_num % k == 0


def _should_keep_epoch_segmented(
    epoch_num: int,
    epoch_seg: int,
    interval_before_seg: int,
    interval_after_seg: int,
) -> bool:
    # 分界点本身始终保留；其余 ckpt 以 epoch_seg 为锚点向前/向后按间隔保留。
    if epoch_num == epoch_seg:
        return True
    if epoch_num < epoch_seg:
        return interval_before_seg > 0 and ((epoch_seg - epoch_num) % interval_before_seg == 0)
    return interval_after_seg > 0 and ((epoch_num - epoch_seg) % interval_after_seg == 0)


def clean_checkpoints(
    directory: str,
    k: Optional[int] = None,
    dry_run: bool = True,
    segmented_cfg: Optional[Dict[str, int]] = None,
):
    """
    清理目录下的 checkpoint。

    规则一：legacy 模式
        - 传入 k
        - 保留 epoch % k == 0 的 ckpt
        - 始终保留最新 ckpt

    规则二：segmented 模式
        - 传入 segmented_cfg={
              'epoch_seg': ...,
              'interval_before_seg': ...,
              'interval_after_seg': ...,
          }
        - epoch < epoch_seg: 以 epoch_seg 为锚点，按 interval_before_seg 间隔保留
        - epoch > epoch_seg: 以 epoch_seg 为锚点，按 interval_after_seg 间隔保留
        - epoch == epoch_seg: 若存在则始终保留
        - 某侧 interval <= 0 时，该侧 ckpt 全部删除
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    segmented_cfg = _normalize_segmented_cfg(segmented_cfg)
    use_segmented_policy = segmented_cfg is not None
    if not use_segmented_policy and (k is None or k <= 0):
        raise ValueError("Legacy cleanup requires keep interval k > 0.")

    epoch_seg = None if segmented_cfg is None else segmented_cfg["epoch_seg"]
    interval_after_seg = None if segmented_cfg is None else segmented_cfg["interval_after_seg"]
    interval_before_seg = None if segmented_cfg is None else segmented_cfg["interval_before_seg"]

    checkpoints = _collect_checkpoints(directory)
    if not checkpoints:
        print("No checkpoint files found matching the pattern 'epoch<number>.pth'.")
        return

    latest_epoch_num = checkpoints[-1][0]
    checkpoints_to_keep: List[Tuple[int, str]] = []
    checkpoints_to_delete: List[Tuple[int, str]] = []

    for epoch_num, filename in checkpoints:
        if use_segmented_policy:
            should_keep = _should_keep_epoch_segmented(
                epoch_num=epoch_num,
                epoch_seg=epoch_seg,
                interval_before_seg=interval_before_seg,
                interval_after_seg=interval_after_seg,
            )
        else:
            should_keep = _should_keep_epoch_legacy(
                epoch_num=epoch_num,
                latest_epoch_num=latest_epoch_num,
                k=k,
            )

        if should_keep:
            checkpoints_to_keep.append((epoch_num, filename))
        else:
            checkpoints_to_delete.append((epoch_num, filename))

    print("-" * 50)
    print(f"Found {len(checkpoints)} total checkpoints in '{directory}'.")
    print(f"Latest checkpoint found: epoch{latest_epoch_num}.pth")
    if use_segmented_policy:
        print(
            "Using segmented policy: "
            f"epoch_seg={epoch_seg}, "
            f"interval_before_seg={interval_before_seg}, "
            f"interval_after_seg={interval_after_seg}"
        )
        print("Keep rule: epoch_seg itself is kept if present; both sides are anchored to epoch_seg.")
        if interval_before_seg <= 0:
            print("Before segment: interval_before_seg <= 0, so all ckpts before epoch_seg will be deleted.")
        if interval_after_seg <= 0:
            print("After segment: interval_after_seg <= 0, so all ckpts after epoch_seg will be deleted.")
    else:
        print(f"Using legacy keep interval k = {k}.")
        print(f"Always keeping the latest checkpoint: epoch{latest_epoch_num}.pth")
    print("-" * 50)

    if dry_run:
        print("DRY RUN MODE: No files will be deleted.")
        print(f"Planning to delete {len(checkpoints_to_delete)} files.")
    else:
        print(f"DELETING {len(checkpoints_to_delete)} files...")
        deleted_count = 0
        for _, filename in checkpoints_to_delete:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"  - Deleted: {filename}")
                deleted_count += 1
            except OSError as exc:
                print(f"  - Error deleting {filename}: {exc}")
        print(f"\nDeletion complete. Deleted {deleted_count} files.")

    print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)

    print(f"\nFiles to keep ({len(checkpoints_to_keep)}):")
    if not checkpoints_to_keep:
        print("  - None")
    else:
        for _, filename in checkpoints_to_keep:
            print(f"  - {filename}")

    action_verb = "to be deleted" if dry_run else "deleted"
    print(f"\nFiles {action_verb} ({len(checkpoints_to_delete)}):")
    if not checkpoints_to_delete:
        print("  - None")
    else:
        for _, filename in checkpoints_to_delete:
            print(f"  - {filename}")

    print("\n" + "=" * 49)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean up checkpoint files. "
            "Use either legacy keep interval k, or segmented policy "
            "(epoch_seg + interval_before_seg + interval_after_seg)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="The directory containing the .pth checkpoint files.",
    )
    parser.add_argument(
        "-k",
        "--keep-interval",
        type=int,
        default=None,
        help="Legacy mode only. Keep epochs that satisfy epoch %% k == 0, and always keep the latest.",
    )
    parser.add_argument(
        "--epoch-seg",
        type=int,
        default=None,
        help="Segment boundary epoch. If set, segmented cleanup mode is enabled.",
    )
    parser.add_argument(
        "--interval-before-seg",
        type=int,
        default=None,
        help="Keep interval for epochs before epoch_seg. <= 0 means delete all before epoch_seg.",
    )
    parser.add_argument(
        "--interval-after-seg",
        type=int,
        default=None,
        help="Keep interval for epochs after epoch_seg. <= 0 means delete all after epoch_seg.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the process without actually deleting files.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.dry_run:
        response = input(
            f"WARNING: You are about to permanently delete files in '{args.directory}'.\n"
            "This action cannot be undone. Are you sure you want to continue? (yes/no): "
        )
        if response.lower() != "yes":
            print("Operation cancelled.")
            return

    clean_checkpoints(
        directory=args.directory,
        k=args.keep_interval,
        dry_run=args.dry_run,
        segmented_cfg=(
            None
            if args.epoch_seg is None
            else {
                "epoch_seg": args.epoch_seg,
                "interval_after_seg": args.interval_after_seg,
                "interval_before_seg": args.interval_before_seg,
            }
        ),
    )


if __name__ == "__main__":
    # main()
    segmented_cfg= {
            "epoch_seg": 30,
            "interval_before_seg": 5,
            "interval_after_seg": 5,
        }
    clean_checkpoints(
        directory = '/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage1_wingtra_interval91_wRejectSampling_infonce_dinov2_adF4_salad',
        k = 10,
        dry_run=False,
        segmented_cfg=None,
    )

