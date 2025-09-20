import os
import torch

def save_param(dirname, dict2save):
    if not os.path.isdir('./exps/' + dirname):
        os.mkdir('./exps/' + dirname)
    epoch_label = dict2save['epoch']
    if isinstance(epoch_label, int):
        save_filename = 'epoch%03d.pth' % epoch_label
    else:
        save_filename = 'epoch%s.pth' % epoch_label
    save_path = os.path.join('./exps', dirname, save_filename)

    torch.save(dict2save, save_path)


def load_param(load_from, dict2loac):
    checkpoint = torch.load(load_from)
    for k, v in dict2loac.items():
        dict2loac[k] = v
        v.load_state_dict(checkpoint[k])
