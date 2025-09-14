import os
import subprocess
import argparse
import sys


def is_git_repo_clean():
    """检查当前的 Git 仓库是否有未提交的修改。"""
    try:
        # --porcelain 是一种为脚本设计的稳定输出格式
        # 如果有任何改动，该命令会有输出；如果工作区是干净的，则无输出。
        output = subprocess.check_output(['git', 'status', '--porcelain']).strip()
        if output:
            return False, "Your working directory has uncommitted changes."
        return True, "Git repository is clean."
    except Exception as e:
        return False, f"Could not check Git status. Are you in a Git repository? Error: {e}"


def parse_git_info(file_path):
    """从 git_info.txt 文件中解析 commit hash 和 patch 内容。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: git_info.txt not found at '{file_path}'")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    commit_hash = None
    patch_content = []
    in_patch_section = False

    for i, line in enumerate(lines):
        if "Commit Hash:" in line and i + 1 < len(lines):
            commit_hash = lines[i + 1].strip()
        elif "Uncommitted Changes (Patch):" in line:
            in_patch_section = True
            # Patch content starts from the line after the header
            patch_content = lines[i + 2:]
            break  # No need to read further after finding the patch section

    # 过滤掉 "Working directory is clean" 的情况
    patch_str = "".join(patch_content).strip()
    if "Working directory is clean" in patch_str:
        patch_str = ""

    return commit_hash, patch_str


def reproduce_code_state(exp_dir):
    # 在 reproduce_code_state 函数的开头
    try:
        # 记录当前分支名或 commit hash
        current_location = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode()
        if current_location == 'HEAD':  # 如果处于分离头指针状态，记录完整的 commit hash
            current_location = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()

        with open('.reproduce_state', 'w') as f:
            f.write(current_location)
    except Exception:
        print("Warning: Could not save the current git state for automatic return.")

    """自动化复现指定实验的代码状态。"""
    print("--- Starting Code Reproduction Process ---")

    # 1. 安全性检查
    print("\n[Step 1/4] Checking repository status...")
    is_clean, message = is_git_repo_clean()
    if not is_clean:
        print(f"Error: Safety check failed. {message}", file=sys.stderr)
        print("Please commit or stash your changes before proceeding.", file=sys.stderr)
        sys.exit(1)  # 退出脚本
    print(f"Success: {message}")

    # 2. 解析 git_info.txt
    print(f"\n[Step 2/4] Parsing git_info.txt from '{exp_dir}'...")
    git_info_file = os.path.join(exp_dir, 'git_info.txt')
    try:
        commit_hash, patch_str = parse_git_info(git_info_file)
        if not commit_hash:
            print(f"Error: Could not find a valid commit hash in '{git_info_file}'.", file=sys.stderr)
            sys.exit(1)
        print(f"  - Found Commit Hash: {commit_hash[:12]}...")
        if patch_str:
            print("  - Found uncommitted changes (patch).")
        else:
            print("  - No uncommitted changes were recorded for this experiment.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 执行 Git Checkout
    print(f"\n[Step 3/4] Checking out to commit '{commit_hash[:12]}...'...")
    try:
        # 使用 subprocess.run, check=True 会在命令失败时抛出异常
        subprocess.run(['git', 'checkout', commit_hash], check=True, capture_output=True)
        print("Success: Git checkout complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error: 'git checkout' failed.", file=sys.stderr)
        print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    # 4. 应用 Patch (如果存在)
    print("\n[Step 4/4] Applying uncommitted changes (if any)...")
    if not patch_str:
        print("No patch to apply. Skipping.")
    else:
        temp_patch_file = 'temp_reproduce.patch'
        try:
            with open(temp_patch_file, 'w') as f:
                f.write(patch_str)

            print(f"  - Applying changes from temporary patch file '{temp_patch_file}'...")
            # 使用 git apply
            subprocess.run(['git', 'apply', temp_patch_file], check=True, capture_output=True)
            print("Success: Patch applied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: 'git apply' failed. The patch may not be applicable.", file=sys.stderr)
            print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
            print("Your repository is now at the specified commit, but the patch failed.", file=sys.stderr)
            print(f"The failed patch has been saved as '{temp_patch_file}' for manual inspection.", file=sys.stderr)
            sys.exit(1)
        finally:
            # 无论成功与否，如果补丁成功应用，最好删除临时文件
            if os.path.exists(temp_patch_file) and 'e' not in locals():
                os.remove(temp_patch_file)

    print("\n--- Code Reproduction Complete ---")
    print("\n✅ Your project is now at the exact code state of the specified experiment.")
    print("Next step: Run your training script using the saved configuration file:")
    print(f"   python your_train_script.py --p_yaml {os.path.join(exp_dir, 'opts.yaml')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Automated script to reproduce the code state of a past experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'exp_dir',
        type=str,
        help="Path to the experiment directory containing git_info.txt and opts.yaml."
    )
    args = parser.parse_args()

    reproduce_code_state(args.exp_dir)

# 外部命令行调用示例
# python util_reproduce_exp.py path_to_the_dir_contains_the_git_info.txt

# 在复现完成exp后，要返回原来的工作区，直接回到上次提交状态即可
# git reset --hard HEAD 的核心是 【清理】：它的目的是在原地不动的情况下，把你的工作区打扫干净。它只关心你当前的位置，不关心你要去哪里。
# git checkout master 的核心是 【移动】：它的目的是把你带到 master 这个分支上去。它关心的是你的目的地，并会把那个目的地的风景（代码状态）展示给你
# git reset --hard HEAD
# git checkout master