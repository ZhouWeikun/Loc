import subprocess
import os


def save_git_info(exp_dir):
    """
    将当前 Git 仓库的 commit hash 和未提交的修改保存到实验目录。
    """
    git_info_path = os.path.join(exp_dir, 'git_info.txt')

    print("正在保存 Git 信息...")

    try:
        # 获取 commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

        # 获取未提交的修改 (diff)
        # --quiet 会在没有 diff 时返回 1，我们用 a || true 来避免报错
        diff = subprocess.check_output('git diff HEAD --stat', shell=True).strip().decode('utf-8')
        uncommitted_changes = subprocess.check_output('git diff HEAD', shell=True).strip().decode('utf-8')

        with open(git_info_path, 'w') as f:
            f.write(f"Commit Hash:\n{commit_hash}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Diff Summary:\n{diff}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Uncommitted Changes (Patch):\n\n{uncommitted_changes}\n")

        print(f"Git 信息已保存到 {git_info_path}")

    except Exception as e:
        print(f"无法获取 Git 信息: {e}")
        print("请确保项目是一个 Git 仓库，并且已安装 Git。")