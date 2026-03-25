import os
import torch
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from shutil import copyfile, copytree, rmtree
import logging
# from train_neuloc.nets_taskflow import mk_vis_encoder
from thop import profile, clever_format
import math
import sys
from pathlib import Path


def get_unique_exp_dir(base_dir: str, exp_name: str) -> str:
    """
    根据基础目录和实验名称，生成一个唯一的实验文件夹路径。

    如果 `base_dir/exp_name` 已存在, 会依次尝试 `exp_name_1`,
    `exp_name_2`, ... 直到找到一个不存在的路径。

    :param base_dir: 实验的根目录路径。
    :param exp_name: 期望的实验文件夹名称。
    :return: 一个唯一的、尚不存在的完整文件夹路径字符串。
    """
    # 将输入的字符串路径转换为Path对象，便于操作
    base_path = Path(base_dir)

    # 1. 首先检查原始名称是否存在
    original_path = base_path / exp_name
    if not original_path.exists():
        # 如果不存在，直接返回原始路径
        # return str(original_path)
        return exp_name

    # 2. 如果原始名称存在，开始尝试添加后缀
    counter = 1
    while True:
        # 构建新的带后缀的名称，例如 "my_experiment_1"
        suffixed_name = f"{exp_name}_{counter}"
        new_path = base_path / suffixed_name

        # 检查带后缀的路径是否存在
        if not new_path.exists():
            # 如果不存在，我们找到了唯一的路径，返回它
            # return str(new_path)
            return suffixed_name

        # 如果仍然存在，增加计数器，进入下一次循环
        counter += 1

def get_logger(log_file='app.log',name='my_app'):
    """
    配置一个专有的、与所有第三方库隔离的应用程序 logger。
    """
    # 定义日志级别和格式
    log_level = logging.INFO
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )

    # 1. 获取我们自己的、有名字的 logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 2. 关键：设置 propagate = False
    #    这会阻止这个 logger 的消息向上传播到被 faiss 等库污染的 root logger。
    #    保证了我们日志的独立性。
    logger.propagate = False

    # 3. 检查并添加 Handlers，防止重复
    if not logger.handlers:
        # 添加文件处理器 (FileHandler)
        fh = logging.FileHandler(log_file, "a")  # 使用追加模式 'a'
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 添加控制台处理器 (StreamHandler)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger

def copy_file_or_tree(path, target_dir):
    target_path = os.path.join(target_dir, path)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path, target_path)
    elif os.path.isfile(path):
        copyfile(path, target_path)

def copyfiles2checkpoints(opt):
    dir_name = os.path.join('gen_fm_exps', 'ckpts', opt.exp_name)
    os.makedirs(dir_name, exist_ok=True)

    # record every run
    copy_file_or_tree('evaluate_retrieval_relrot_wingtra.py', dir_name)

    # copy_file_or_tree('test.py', dir_name),about eval
    copy_file_or_tree('evaluate_dsalad.py', dir_name)
    copy_file_or_tree('evaluate_filter_inference.py', dir_name)
    copy_file_or_tree('evaluate_retrieval.py', dir_name)
    copy_file_or_tree('evaluate_retrieval_relrot.py', dir_name)
    copy_file_or_tree('util_circorr_fm_radon.py', dir_name)

    # about dirs:
    copy_file_or_tree('datasets_custom', dir_name)
    copy_file_or_tree('losses', dir_name)
    copy_file_or_tree('models', dir_name)
    copy_file_or_tree('optimizers', dir_name)
    copy_file_or_tree('tool', dir_name)

    # save opts
    grouped_params = {}
    for group, params in opt.group_dict.items():
        grouped_params[group] = {param: getattr(opt, param) for param in params}
    with open('%s/opts_wingtra.yaml' % dir_name, 'w') as fp:
        yaml.dump(grouped_params, fp, default_flow_style=False)


def copyfiles2checkpoints_map_learner(opt):
    base_dir = getattr(opt, 'dir2save_ckpt', None) or getattr(opt, 'exps_dir', 'gen_fm_exps/ckpts')
    dir_name = os.path.join(base_dir, opt.exp_name)
    os.makedirs(dir_name, exist_ok=True)

    # 1. 使用 vars() 将 Namespace 对象转换为字典
    args_dict = vars(opt)
    # save opts
    # grouped_params = {}
    # for group, params in opt.group_dict.items():
    #     grouped_params[group] = {param: getattr(opt, param) for param in params}
    with open('%s/args.yaml' % dir_name, 'w') as fp:
        yaml.dump(args_dict, fp, default_flow_style=False)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
# ---------------------------


def save_network(network, dirname, epoch_label):
    save_dir = os.path.join('./gen_fm_exps/ckpts', dirname)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(epoch_label, int):
        save_filename = 'epoch%03d.pth' % epoch_label
    else:
        save_filename = 'epoch%s.pth' % epoch_label
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()

def save_param(dirname, dict2save,epoch_label):
    save_dir = os.path.join('./gen_fm_exps/ckpts', dirname)
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(epoch_label, int):
        save_filename = 'epoch%03d.pth' % epoch_label
    else:
        save_filename = 'epoch%s.pth' % epoch_label
    save_path = os.path.join(save_dir, save_filename)

    torch.save(dict2save, save_path)

def load_network_wstate(load_from,network,optimizer,lr_scheduler):
    checkpoint = torch.load(load_from)

    if 'model_state' in checkpoint.keys():
        network.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        current_epoch = checkpoint["epoch"]
    else:
        network.load_state_dict(checkpoint)
        current_epoch=0

    # torch.save({
    #     "model_state": network.cpu().state_dict(),
    #     "optimizer_state": optimizer.state_dict(),
    #     "scheduler_state": lr_scheduler.state_dict(),
    #     "epoch": epoch_label,
    # }, save_path)
    return current_epoch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def check_box(images, boxes):
    # Unorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # images = Unorm(images)*255
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    boxes = (boxes.cpu().detach().numpy()/16*255).astype(np.int)
    for img, box in zip(images, boxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        rect = plt.Rectangle(box[0:2], box[2]-box[0], box[3]-box[1])
        ax.add_patch(rect)
        plt.show()


######################################################################
#  Load model for resume
# ---------------------------
def load_network(opt):
    save_filename = opt.checkpoint
    model = make_img_encoder(opt)
    # print('Load the model from %s' % save_filename)
    network = model
    network.load_state_dict(torch.load(save_filename))
    return network




def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)


def get_preds(outputs, outputs2):
    if isinstance(outputs, list):
        preds = []
        preds2 = []
        for out, out2 in zip(outputs, outputs2):
            preds.append(torch.max(out.data, 1)[1])
            preds2.append(torch.max(out2.data, 1)[1])
    else:
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
    return preds, preds2


def calc_flops_params(model,
                      input_size_drone,
                      input_size_satellite,
                      ):
    inputs_drone = torch.randn(input_size_drone).cuda()
    inputs_satellite = torch.randn(input_size_satellite).cuda()
    total_ops, total_params = profile(
        model, (inputs_drone, inputs_satellite,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)


def Distance(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a)
                                  * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


LOG_FILENAME = "gen_fm_exps/logs/debug/train_w_uavre.log"
def run_test():
    print("--- 开始日志调试 ---")

    # 诊断文件路径和权限
    log_dir = os.path.dirname(os.path.abspath(LOG_FILENAME))
    print(f"日志文件的绝对路径: {os.path.abspath(LOG_FILENAME)}")
    print(f"日志文件所在的目录: {log_dir}")
    print(f"目录是否存在? {os.path.exists(log_dir)}")
    print(f"目录是否可写? {os.access(log_dir, os.W_OK)}")

    # 获取 logger 实例
    # 使用一个独立的名字 'debug_app' 以免和主程序冲突
    logger = get_logger(LOG_FILENAME, name='debug_app')

    print("\n--- 准备写入日志 ---")
    logger.info("这是一条来自【调试脚本】的 INFO 消息。")
    logger.warning("这是一条来自【调试脚本】的 WARNING 消息。")
    print("--- 日志写入已调用 ---")

    # 强制将所有缓冲区的日志刷新到文件中
    # logging.shutdown()
    # print("\n--- logging.shutdown() 已调用，强制刷新缓冲区 ---")

    # 检查文件内容
    try:
        with open(LOG_FILENAME, 'r') as f:
            content = f.read()
            if "调试脚本" in content:
                print("\n✅ 成功！调试消息已写入文件。")
            else:
                print("\n❌ 失败！文件中未找到调试消息。")
    except FileNotFoundError:
        print("\n❌ 失败！日志文件未被创建。")
    except Exception as e:
        print(f"\n❌ 读取文件时发生错误: {e}")

if __name__ == '__main__':
    run_test()
