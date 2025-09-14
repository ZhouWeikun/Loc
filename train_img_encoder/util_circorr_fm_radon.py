import numpy as np
import torch
from torch_radon import Radon

class RadonHandler(object):
    def __init__(self,n_line=14,delta_deg=5,normscale=0.1,device=None,input_hw=(14,14)):
        """
        input_hw: assuming that the forward_func receives the featmap in the shape of [b,c,h,w]
        """
        super(RadonHandler, self).__init__()
        self.delta_deg = delta_deg
        radon_degs = np.array([delta_deg * i for i in range(int(360 / delta_deg))])
        radon_degs = np.deg2rad(radon_degs)
        self.radon = Radon(n_line, radon_degs)
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        self.sg_normer = torch.ones(1, 1, input_hw[0],input_hw[1]).to(self.device)
        with torch.no_grad():
            self.sg_normer = self.radon.forward(self.sg_normer)*normscale
        """
            # ## normalize the sg by the length of line integration
            # img_weight = torch.ones(1,1,*featmaps.shape[1:-1]).cuda()
            # sg_weight = radon.forward(img_weight).squeeze()
            # sgfeats_normed = sgfeats/(sg_weight[None,...]*0.1) if len(sgfeats.shape)==3 else sgfeats/(sg_weight[None,None,...]*0.1)
            # sgfeats = sgfeats_normed
        """

    def forward(self, featmap:torch.Tensor,norm_manner='l2'):
        """the shape of featmap should be [b,c,h,w]"""
        sg = self.radon.forward(featmap)/self.sg_normer
        sg  = sg / sg.sum(dim=(1, 2, 3), keepdim=True)

        if norm_manner =="sum":
            sg = sg / sg.sum(dim=(1, 2, 3), keepdim=True)
        elif norm_manner=='l2':
            norm = sg.norm(p=2, dim=(1, 2, 3), keepdim=True)
            sg = sg / (norm + 1e-12)

        return sg

def circorr_fm_radon(sg_query:torch.Tensor,sg_ref:torch.Tensor):
    """ verison 1
    cir = integrate( f(t)*g(t+u) )dt；
    sg_query=sg_rotated=ft,sg_ref=sg_wo_rotated=gt
    both sg_q and sg_r are in the shape of [B,C,H,W]
    assuming that this func calculates the correlation between sg_queries derived from a series of counterclockwise rotations of sg_query and sg_ref
    """
    x1 = torch.nn.functional.normalize(sg_query, dim=(-1))
    x2 = torch.nn.functional.normalize(sg_ref, dim=(-1))

    a_fft = torch.fft.fft2(x1, dim=(-2), norm="ortho")
    b_fft = torch.fft.fft2(x2, dim=(-2), norm="ortho")

    corr = torch.fft.ifft2(a_fft * b_fft.conj(), dim=-2, norm="ortho")  # compute the corr in the dim of -2
    corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
    # sum the correlation over the channels
    corr = torch.sum(corr, dim=-3)
    # sum the correlation over the width
    corr = torch.sum(corr, dim=-1)

    return corr

# def circorr_fm_radon(sg_ref:torch.Tensor,sg_query:torch.Tensor):
#     """version 0
#     cir = integrate( f(t)*g(t+u) )dt
#     both sg_q and sg_r are in the shape of [B,C,H,W]
#     assuming that this func calculates the correlation between sg_queries derived from a series of counterclockwise rotations of sg_query and sg_ref
#     """
#     x1 = torch.nn.functional.normalize(sg_query, dim=(-1))
#     x2 = torch.nn.functional.normalize(sg_ref, dim=(-1))
#
#     a_fft = torch.fft.fft2(x1, dim=(-2), norm="ortho")
#     b_fft = torch.fft.fft2(x2, dim=(-2), norm="ortho")
#
#     corr = torch.fft.ifft2(a_fft * b_fft.conj(), dim=-2, norm="ortho")  # compute the corr in the dim of -2
#     corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2 + 1e-15)
#     # sum the correlation over the channels
#     corr = torch.sum(corr, dim=-3)
#     # sum the correlation over the width
#     corr = torch.sum(corr, dim=-1)
#
#     return corr

def norm_rot(rotdeg_fm_north_anticlock_positive):
    """
    rotdeg_fm_north_anticlock_positive: the deg that in [-pi,pi],
    convert deg in [-pi,pi] to [0,360]
    """
    relrot = []
    for r in rotdeg_fm_north_anticlock_positive:
        if r < 0:
            relrot.append(360+r)
        else:
            relrot.append(r)
    relrot = np.array(relrot)
    return relrot