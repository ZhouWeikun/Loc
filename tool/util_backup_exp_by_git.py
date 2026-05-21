import subprocess
import os
import yaml

def backup_experiment(exp_dir,opt=None):
    """
    将当前 Git 仓库的 commit hash 和未提交的修改保存到实验目录。
    """
    if not exp_dir or not os.path.isdir(exp_dir):
        print(f"跳过实验备份：目录不存在 {exp_dir}")
        return

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

    if opt is not None:  # save opts
        # 保存完整配置：包括所有group_dict中的参数 + scenes_setting
        grouped_params = {}

        # checkpoint参数列表（用于过滤空值）
        checkpoint_params = ['load2train', 'load2test', 'load_stage1_ckpt', 'load_stage2_ckpt']

        # 1. 保存group_dict中定义的所有参数
        for group, params in opt.group_dict.items():
            grouped_params[group] = {}
            for param in params:
                if hasattr(opt, param):
                    value = getattr(opt, param)
                    # 对于checkpoint参数，只保存非空值
                    if param in checkpoint_params:
                        if value and value != "":  # 只保存有值的checkpoint参数
                            grouped_params[group][param] = value
                    else:
                        # 非checkpoint参数，全部保存
                        grouped_params[group][param] = value

        # 2. 添加scenes_setting（如果存在）
        # scenes_setting包含了完整的场景配置，包括采样策略
        if hasattr(opt, 'scenes_setting'):
            grouped_params['scenes_setting'] = opt.scenes_setting

        with open(f'{exp_dir}/opts.yaml', 'w') as fp:
            yaml.dump(grouped_params, fp, default_flow_style=False)

        print(f"配置已保存到 {exp_dir}/opts.yaml")
