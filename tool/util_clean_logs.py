import os
import shutil
import argparse
from typing import List, Set


def _list_subdirs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    )


def _collect_ckpt_exp_names(
    ckpt_root: str,
    require_checkpoint_file: bool = False,
) -> Set[str]:
    exp_names: Set[str] = set()
    for exp_name in _list_subdirs(ckpt_root):
        ckpt_dir = os.path.join(ckpt_root, exp_name)
        if not require_checkpoint_file:
            exp_names.add(exp_name)
            continue

        has_ckpt_file = any(
            filename.endswith(".pth")
            and os.path.isfile(os.path.join(ckpt_dir, filename))
            for filename in os.listdir(ckpt_dir)
        )
        if has_ckpt_file:
            exp_names.add(exp_name)
    return exp_names


def clean_logs_without_ckpt(
    ckpt_root: str,
    log_root: str,
    dry_run: bool = True,
    require_checkpoint_file: bool = False,
) -> List[str]:
    """
    删除 logs 根目录下那些在 ckpts 根目录中没有对应实验目录的日志文件夹。

    Args:
        ckpt_root: checkpoint 根目录。
        log_root: log 根目录。
        dry_run: True 时只打印，不实际删除。
        require_checkpoint_file: True 时要求对应 ckpt 目录下至少存在一个 .pth 文件，
            才认为该实验“有效存在”。

    Returns:
        被判定为无用、需要删除的 log 子目录名称列表。
    """
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"Checkpoint root not found: '{ckpt_root}'")
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Log root not found: '{log_root}'")

    ckpt_exp_names = _collect_ckpt_exp_names(
        ckpt_root,
        require_checkpoint_file=require_checkpoint_file,
    )
    log_exp_names = _list_subdirs(log_root)

    logs_to_delete = [
        exp_name for exp_name in log_exp_names
        if exp_name not in ckpt_exp_names
    ]

    print("-" * 60)
    print(f"Checkpoint root: {ckpt_root}")
    print(f"Log root: {log_root}")
    print(f"Found {len(ckpt_exp_names)} valid ckpt experiment folders.")
    print(f"Found {len(log_exp_names)} log folders.")
    print(f"Unmatched log folders: {len(logs_to_delete)}")
    print("-" * 60)

    if not logs_to_delete:
        print("No useless log folders found.")
        return []

    if dry_run:
        print("DRY RUN MODE: No folders will be deleted.")
    else:
        print("Deleting unmatched log folders...")

    for exp_name in logs_to_delete:
        log_dir = os.path.join(log_root, exp_name)
        if dry_run:
            print(f"  - Would delete: {log_dir}")
            continue
        shutil.rmtree(log_dir)
        print(f"  - Deleted: {log_dir}")

    print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
    action_verb = "to be deleted" if dry_run else "deleted"
    print(f"Log folders {action_verb} ({len(logs_to_delete)}):")
    for exp_name in logs_to_delete:
        print(f"  - {exp_name}")
    print("=" * 49)

    return logs_to_delete


if __name__ == "__main__":
    # clean manually:
    ckpt_root = '/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts'
    log_dir = '/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/logs'
    clean_logs_without_ckpt(
        ckpt_root=ckpt_root,
        log_root=log_dir,
        dry_run=False,
        require_checkpoint_file=True,
    )

    # parser = argparse.ArgumentParser(
    #     description=(
    #         "Clean log folders that do not have a corresponding experiment "
    #         "folder under the checkpoint root."
    #     )
    # )
    # parser.add_argument(
    #     "--ckpt-root",
    #     type=str,
    #     # required=True,
    #     help="Root directory that stores checkpoint experiment folders.",
    # )
    # parser.add_argument(
    #     "--log-root",
    #     type=str,
    #     # required=True,
    #     help="Root directory that stores log experiment folders.",
    # )
    # parser.add_argument(
    #     "--execute",
    #     action="store_true",
    #     help="Actually delete unmatched log folders. Default is dry-run.",
    # )
    # parser.add_argument(
    #     "--require-checkpoint-file",
    #     action="store_true",
    #     help="Only treat ckpt folders with at least one .pth file as valid.",
    # )
    #
    # args = parser.parse_args()
    # clean_logs_without_ckpt(
    #     ckpt_root=args.ckpt_root,
    #     log_root=args.log_root,
    #     dry_run=not args.execute,
    #     require_checkpoint_file=args.require_checkpoint_file,
    # )
    #
