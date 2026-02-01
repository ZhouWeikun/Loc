OPTIMIZER_TEMPLATES = {
    'sgd': {
        'type': 'sgd',
        'weight_decay': 5e-4,   # 全局默认（可被 param_groups 覆盖）
        'momentum': 0.9,
        'nesterov': True,
        'param_groups': [
            # 专门放 bias / norm 的 no_decay
            {'param_source': 'no_decay', 'lr': 1e-3, 'weight_decay': 0.0},
            # 预训练 encoder：微调用非常小 lr，no_decay 分组在下方实施
            {'param_source': 'img_encoder', 'lr': 1e-3, 'weight_decay': 5e-4},
            # mlp：从头训练，适度正则化以防过拟合
            {'param_source': 'mlp', 'lr': 1e-2, 'weight_decay': 1e-4},
            # hash_grid：对 embedding 做轻微 L2 正则，避免完全为 0，但不要太大
            {'param_source': 'grid', 'lr': 5e-3, 'weight_decay': 1e-6},
        ]
    },
    'adamw': {
        'type': 'adamw',
        'weight_decay': 1e-2,   # 全局默认，可在 group 中覆盖
        'eps': 1e-8,
        'betas': (0.9, 0.999),
        'param_groups': [
            {'param_source': 'no_decay', 'lr': 5e-5, 'weight_decay': 0.0},
            # 微调 encoder：非常小的 lr 与适度 weight_decay（可用 1e-2 或更小）
            {'param_source': 'img_encoder', 'lr': 5e-5, 'weight_decay': 1e-2},
            # mlp：用更强的正则（防止从头训练的 MLP 过拟合）
            {'param_source': 'mlp', 'lr': 1e-3, 'weight_decay': 1e-4},
            # hash_grid：轻度 L2（若观测到 embedding 被压扁或性能下降，降到 1e-7 或 0）
            {'param_source': 'grid', 'lr': 5e-2, 'weight_decay': 1e-6},
        ]
    },
    'adam': {
        'type': 'adam',
        'weight_decay': 1e-6,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'param_groups': [
            {'param_source': 'no_decay', 'lr': 5e-5, 'weight_decay': 0.0},
            {'param_source': 'img_encoder', 'lr': 5e-5, 'weight_decay': 1e-6},
            {'param_source': 'vis_encoder', 'lr': 5e-5, 'weight_decay': 1e-6},
            {'param_source': 'aggregator', 'lr': 1e-4, 'weight_decay': 1e-6},
            {'param_source': 'vis_aggregator', 'lr': 1e-4, 'weight_decay': 1e-6},
            {'param_source': 'grid', 'lr': 1, 'weight_decay': 0},
            {'param_source': 'grid_mlp', 'lr': 1e-2, 'weight_decay': 1e-6},
            {'param_source': 'metric_net', 'lr': 1e-3, 'weight_decay': 1e-6},
            {'param_source': 'projector', 'lr': 1e-4, 'weight_decay': 1e-6},
            {'param_source': 'rank_former', 'lr': 1e-3, 'weight_decay': 1e-6},
            {'param_source': 'loss_fn', 'lr': 1e-3,'weight_decay': 0.0}
        ]
    }
}


from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

def _is_norm_or_bias_name(name: str) -> bool:
    n = name.lower()
    return ('bias' in n) or ('bn' in n) or ('layernorm' in n) or ('ln' in n) or ('norm' in n) or ('batchnorm' in n)

def create_optimizer_w_temple(modules: dict[str, nn.Module], opt_template_name: str) -> optim.Optimizer:
    """
    Robust builder that prevents duplicate parameters across param-groups.
    - modules: {'img_encoder': ..., 'mlp': ..., 'hash_grid': ...}
    - OPTIMIZER_TEMPLATES must contain a template for opt_template_name
    - Supports 'no_decay' and 'no_decay.<module>' param_source
    """
    name = opt_template_name.lower()
    if name not in OPTIMIZER_TEMPLATES:
        raise ValueError(f"Optimizer template '{name}' not found in OPTIMIZER_TEMPLATES.")

    cfg = deepcopy(OPTIMIZER_TEMPLATES[name])
    global_wd = cfg.get('weight_decay', 0.0)

    # Build a mapping of "module_name.param_name" -> parameter
    named_params = {}
    for module_name, module in modules.items():
        for n, p in module.named_parameters(recurse=True):
            named_params[f"{module_name}.{n}"] = p

    param_groups = []
    assigned_param_ids = set()  # track id(p) to avoid duplicates
    total_added = 0
    total_skipped_duplicates = 0

    print(f"Building optimizer from template '{name}':")
    for g in cfg.get('param_groups', []):
        src = g.get('param_source')
        lr = g.get('lr')
        if src is None or lr is None:
            continue
        group_wd = g.get('weight_decay', global_wd)

        # handle no_decay (global)
        if src == 'no_decay':
            group_params = []
            for full_name, p in named_params.items():
                if not p.requires_grad:
                    continue
                # heuristic: check only the parameter's short name (after first dot)
                short_name = full_name.split('.', 1)[1] if '.' in full_name else full_name
                if _is_norm_or_bias_name(short_name):
                    pid = id(p)
                    if pid in assigned_param_ids:
                        total_skipped_duplicates += 1
                        continue
                    group_params.append(p)
                    assigned_param_ids.add(pid)
                    total_added += 1
            if group_params:
                param_groups.append({'params': group_params, 'lr': lr, 'weight_decay': 0.0})
                print(f"  - no_decay (global): {len(group_params)} params, lr={lr:.2e}, wd=0.0")
            else:
                print("  - no_decay (global): matched 0 params.")
            continue

        # handle no_decay.<module>
        if src.startswith('no_decay.'):
            _, mod_name = src.split('.', 1)
            if mod_name not in modules:
                print(f"  - warning: module '{mod_name}' not found for '{src}', skipping.")
                continue
            group_params = []
            prefix = f"{mod_name}."
            for full_name, p in named_params.items():
                if not p.requires_grad:
                    continue
                if not full_name.startswith(prefix):
                    continue
                short_name = full_name[len(prefix):]
                if _is_norm_or_bias_name(short_name):
                    pid = id(p)
                    if pid in assigned_param_ids:
                        total_skipped_duplicates += 1
                        continue
                    group_params.append(p)
                    assigned_param_ids.add(pid)
                    total_added += 1
            if group_params:
                param_groups.append({'params': group_params, 'lr': lr, 'weight_decay': 0.0})
                print(f"  - {src}: {len(group_params)} params, lr={lr:.2e}, wd=0.0")
            else:
                print(f"  - {src}: matched 0 params.")
            continue

        # normal module group (e.g., 'img_encoder', 'mlp', 'hash_grid')
        if src not in modules:
            print(f"  - warning: module '{src}' not found in provided dict, skipping.")
            continue

        group_params = []
        for p in modules[src].parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in assigned_param_ids:
                total_skipped_duplicates += 1
                continue
            group_params.append(p)
            assigned_param_ids.add(pid)
            total_added += 1

        if group_params:
            param_groups.append({'params': group_params, 'lr': lr, 'weight_decay': group_wd})
            print(f"  - {src}: {len(group_params)} params, lr={lr:.2e}, wd={group_wd:.2e}")
        else:
            print(f"  - {src}: no trainable params added (maybe all were already assigned or frozen).")

    print(f"Total params added: {total_added}, duplicates skipped: {total_skipped_duplicates}")

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters collected for optimizer; check your template module names.")

    # instantiate optimizer
    if name == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=cfg.get('momentum', 0.9),
                              nesterov=cfg.get('nesterov', False))
    elif name == 'adam':
        optimizer = optim.Adam(param_groups, betas=cfg.get('betas', (0.9, 0.999)), eps=cfg.get('eps', 1e-8))
    elif name == 'adamw':
        optimizer = optim.AdamW(param_groups, betas=cfg.get('betas', (0.9, 0.999)), eps=cfg.get('eps', 1e-8))
    else:
        raise ValueError(f"Unsupported optimizer '{name}'.")

    return optimizer


###############org#################
# todo:lr_scheduler
from torch.optim import lr_scheduler
def make_optimizer(model,opt):
    backbone_params = []
    backbone_params += list(map(id, model.backbone.parameters()))
    extra_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    base_params = filter(lambda p: id(p) in backbone_params, model.parameters())

    if opt.optimizer['name'].lower() == 'sgd':
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.3 * opt.optimizer['lr']},
            {'params': extra_params, 'lr': opt.optimizer['lr']}
        ], weight_decay=opt.optimizer['weight_decay'], momentum=opt.optimizer['momentum'], nesterov=True)

    if opt.lr_sched['name'].lower() == 'multistep':
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_sched['milestones'], gamma=opt.lr_sched['gamma'])

    return optimizer,exp_lr_scheduler


"""
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, #0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        }
"""
# configure the optimizer
def configure_optimizers(self):
    if self.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
    elif self.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    elif self.optimizer.lower() == 'adam':
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    else:
        raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

    if self.lr_sched.lower() == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'],
                                             gamma=self.lr_sched_args['gamma'])
    elif self.lr_sched.lower() == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
    elif self.lr_sched.lower() == 'linear':
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.lr_sched_args['start_factor'],
            end_factor=self.lr_sched_args['end_factor'],
            total_iters=self.lr_sched_args['total_iters']
        )
    return [optimizer], [scheduler]


# configure the optizer step, takes into account the warmup stage
def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    # warm up lr
    optimizer.step(closure=optimizer_closure)
    self.lr_schedulers().step()