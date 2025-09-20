import torch.optim as optim
from torch.optim import lr_scheduler
import torch


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