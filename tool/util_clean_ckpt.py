import os
import re
import argparse
from typing import List, Tuple


def clean_checkpoints(directory: str, k: int, dry_run: bool = True):
    """
    Cleans up checkpoint files in a directory, keeping one every k epochs.

    Args:
        directory (str): The path to the directory containing the checkpoints.
        k (int): The interval for keeping checkpoints. For example, if k=10,
                 it will keep epochs that are multiples of 10.
        dry_run (bool): If True, only prints the files that would be deleted
                        without actually deleting them.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    # 正则表达式，用于从 'epoch3279.pth' 这样的文件名中提取数字 3279
    # \d+ 匹配一个或多个数字
    pattern = re.compile(r'epoch(\d+)\.pth')

    checkpoints: List[Tuple[int, str]] = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # match.group(1) 会获取第一个括号内的内容，即 epoch 的数字
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, filename))

    if not checkpoints:
        print("No checkpoint files found matching the pattern 'epoch<number>.pth'.")
        return

    # 按 epoch 数字从小到大排序
    checkpoints.sort(key=lambda x: x[0])

    # 找出最新的 epoch 编号，这个文件我们总是保留
    latest_epoch_num = checkpoints[-1][0] if checkpoints else -1

    files_to_delete = []
    files_to_keep = []

    for epoch_num, filename in checkpoints:
        # 检查是否要保留
        # 保留的条件:
        # 1. 它是最新的一个 epoch
        # 2. 它的 epoch 编号是 k 的整数倍
        if epoch_num == latest_epoch_num or epoch_num % k == 0:
            files_to_keep.append(filename)
        else:
            files_to_delete.append(filename)

    print("-" * 50)
    print(f"Found {len(checkpoints)} total checkpoints in '{directory}'.")
    print(f"Keep interval k = {k}.")
    print(f"Always keeping the latest checkpoint: epoch{latest_epoch_num}.pth")
    print("-" * 50)

    if dry_run:
        print("DRY RUN MODE: No files will be deleted.")
        print(f"Planning to delete {len(files_to_delete)} files.")
    else:
        print(f"DELETING {len(files_to_delete)} files...")
        deleted_count = 0
        for filename in files_to_delete:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"  - Deleted: {filename}")
                deleted_count += 1
            except OSError as e:
                print(f"  - Error deleting {filename}: {e}")
        print(f"\nDeletion complete. Deleted {deleted_count} files.")

    # --- 主要修改部分 ---
    # 无论是否为 dry_run，都在最后清晰地打印总结报告
    print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)

    # 打印保留的文件列表
    print(f"\nFiles to keep ({len(files_to_keep)}):")
    if not files_to_keep:
        print("  - None")
    else:
        for f in sorted(files_to_keep):
            print(f"  - {f}")

    # 打印将要删除或已经删除的文件列表
    action_verb = "to be deleted" if dry_run else "deleted"
    print(f"\nFiles {action_verb} ({len(files_to_delete)}):")
    if not files_to_delete:
        print("  - None")
    else:
        for f in sorted(files_to_delete):
            print(f"  - {f}")

    print("\n" + "=" * 49)


if __name__ == "__main__":
    # 使用 argparse 来处理命令行参数，使脚本更易用
    # parser = argparse.ArgumentParser(
    #     description="Clean up old checkpoint files, keeping one every k epochs and always the latest one.",
    #     formatter_class=argparse.RawTextHelpFormatter
    # )
    #
    # parser.add_argument(
    #     '-d', '--directory',
    #     type=str,
    #     required=True,
    #     help="The directory containing the .pth checkpoint files."
    # )
    #
    # parser.add_argument(
    #     '-k', '--keep-interval',
    #     type=int,
    #     required=True,
    #     help="The interval of epochs to keep. E.g., 100 means keep epoch100, epoch200, etc."
    # )
    #
    # parser.add_argument(
    #     '--dry-run',
    #     action='store_true',
    #     help="Simulate the process without actually deleting any files. Highly recommended for the first run."
    # )
    #
    # args = parser.parse_args()
    #
    # # 确认执行删除操作
    # if not args.dry_run:
    #     response = input(
    #         f"WARNING: You are about to permanently delete files in '{args.directory}'.\n"
    #         "This action cannot be undone. Are you sure you want to continue? (yes/no): "
    #     )
    #     if response.lower() != 'yes':
    #         print("Operation cancelled.")
    #         exit()

    # clean_checkpoints(args.directory, args.keep_interval, args.dry_run)
    #todo:先确定最优ckpt后再清理
    clean_checkpoints('/home/data/zwk/pyproj_neuloc_v0/gen_fm_exps/ckpts/stage2_visloc03_tripleLoss_singleEdge_hardest_fm_mask_codebookW19_mlpH1024B1_PN1cubie'
                      , 50, False)