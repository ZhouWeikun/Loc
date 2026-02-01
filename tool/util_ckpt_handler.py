import os
import torch

# def save_param(dirname, dict2save):
#     if not os.path.isdir('./exps/' + dirname):
#         os.mkdir('./exps/' + dirname)
#     epoch_label = dict2save['epoch']
#     if isinstance(epoch_label, int):
#         save_filename = 'epoch%03d.pth' % epoch_label
#     else:
#         save_filename = 'epoch%s.pth' % epoch_label
#     save_path = os.path.join('./exps', dirname, save_filename)
#
#     torch.save(dict2save, save_path)
def save_param(dirname, dict2save):
    """
    保存参数字典到文件。
    该函数会自动处理字典中的 PyTorch 模块，调用 .state_dict() 方法。
    """
    if not os.path.isdir('./exps/'):
        os.mkdir('./exps/')
    if not os.path.isdir('./exps/' + dirname):
        os.mkdir('./exps/' + dirname)

    # 1. 准备要保存的 state_dict 字典
    state_dict_to_save = {}
    for key, value in dict2save.items():
        # 判断 value 是否是 PyTorch 模块或优化器等拥有 state_dict 的对象
        if hasattr(value, 'state_dict'):
            # 如果是，则调用 .state_dict() 方法
            state_dict_to_save[key] = value.state_dict()
        else:
            # 如果不是（例如 epoch, learning_rate 等元数据），则直接保留原值
            state_dict_to_save[key] = value

    # 2. 处理文件名和路径 (这部分逻辑与您原先的一致)
    if 'epoch' not in state_dict_to_save:
        print("警告: 字典中缺少 'epoch' 信息，将使用 'latest'作为文件名。")
        epoch_label = 'latest'
    else:
        epoch_label = state_dict_to_save['epoch']

    if isinstance(epoch_label, int):
        save_filename = 'epoch%03d.pth' % epoch_label
    else:
        save_filename = 'epoch%s.pth' % epoch_label

    save_path = os.path.join('./exps', dirname, save_filename)

    # 3. 保存处理后的 state_dict 字典
    torch.save(state_dict_to_save, save_path)
    print(f"参数已成功保存到: {save_path}")



# def load_param(load_from, dict2load):
#     checkpoint = torch.load(load_from)
#     for k, v in dict2load.items():
#         if k == 'epoch':
#             dict2load['epoch'] = checkpoint['epoch']
#         else:
#             dict2load[k] = v
#             v.load_state_dict(checkpoint[k])

# --- 修正后的函数 ---
def load_param(load_from, dict2load):
    print(f"Loading parameters from: {load_from}")
    checkpoint = torch.load(load_from, map_location=lambda storage, loc: storage) # 保证设备兼容性

    for k, v in dict2load.items():
        if k in checkpoint:
            if k == 'epoch':
                dict2load['epoch'] = checkpoint['epoch']
                print(f"Loaded epoch: {dict2load['epoch']}")
            else:
                # 直接就地加载参数，不需要 "dict2load[k] = v"
                if hasattr(v, 'load_state_dict'): # 确保对象有这个方法
                    v.load_state_dict(checkpoint[k])
                    print(f"Loaded parameters for: {k}")
                else:
                    # 如果v不是模型或优化器，只是一个普通变量，直接赋值
                    dict2load[k] = checkpoint[k]
        else:
            print(f"Warning: Key '{k}' not found in checkpoint file.")

    return checkpoint