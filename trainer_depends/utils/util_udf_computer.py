import math
import torch

class EuclideanNormalized(object):
    def __init__(self, sat_dataset):
        self.sat_dataset = sat_dataset

class UDFComputer(object):
    def __init__(self, sat_dataset):
        self.sat_dataset = sat_dataset

        # 定义距离归一化因子 (根据实验调整)
        self.norm_factor_rc = math.sqrt(self.sat_dataset.nr2sample_h ** 2 + self.sat_dataset.nc2sample_w ** 2)
        self.nrom_factor_rot = torch.pi  # todo:make the threshold auto
        self.nrom_factor_scale = math.log( self.sat_dataset.satimgsize_scale_to_refm_boundary[1] / self.sat_dataset.satimgsize_scale_to_refm_boundary[0])

        # weight definity,version1:
        self.w_rc = 0.6  # 位置权重，通常设为1.0作为基准
        self.w_r = 0.3 # 位置权重，通常设为1.0作为基准
        self.w_s = 0.1  # 尺度权重
        rc_dist_threshold_accpetable = self.sat_dataset.halfimg_radius_nrc
        # self.neg_weight_fm_nrc_dist = lambda x:torch.sigmoid(10*x/rc_dist_threshold_accpetable-5) #weight in [0,1],assuming x/rc_dist_threshold_accpetable mapping rc_dist_threshold_accpetable to 1
        # self.neg_weight_fm_nrc_dist = lambda x:torch.sigmoid(7.5*x/rc_dist_threshold_accpetable-3.75) #x/rc_dist_threshold_accpetable -> mapping rc_dist_threshold_accpetable to 1
        self.neg_weight_fm_nrc_dist = lambda x:torch.sigmoid(8.5*x/rc_dist_threshold_accpetable-4.68) #x/rc_dist_threshold_accpetable -> mapping rc_dist_threshold_accpetable to 1
        self.udf_threshold_accpetable = rc_dist_threshold_accpetable

    def compute_udf(self,q_coords_flatten, ref_coords_flatten):
        rc_dist_mat = q_coords_flatten[:, :2].unsqueeze(1) - ref_coords_flatten[:, :2].unsqueeze(0)
        rc_dist_mat = torch.norm(rc_dist_mat, dim=-1)
        r_dist_mat = q_coords_flatten[:, 2].unsqueeze(1) - ref_coords_flatten[:, 2].unsqueeze(0)
        r_dist_mat = torch.abs(torch.atan2(torch.sin(r_dist_mat), torch.cos(r_dist_mat))).squeeze()  # atan2 函数的输出范围是 [-π, π]
        s_dist_mat = q_coords_flatten[:, 3].unsqueeze(1) / ref_coords_flatten[:, 3].unsqueeze(0)
        s_dist_mat = torch.abs(torch.log(s_dist_mat)).squeeze()  # 比例关系在对数空间中会变为加减关系
        udf_dist_mat = self.compute_udf_fm_diff(rc_dist_mat, r_dist_mat, s_dist_mat)
        # positive_mat = udf_dist_mat < self.sat_dataset.halfimg_radius_nrc
        return udf_dist_mat

    def compute_udf_fm_diff(self, dists_rc, dists_rot, dists_scale=None):
        dists_rc_normed = dists_rc / self.norm_factor_rc
        dists_rot_normed = dists_rot / self.nrom_factor_rot
        dists_scale_normed = dists_scale / self.nrom_factor_scale if dists_scale is not None else None

        # 计算加权的平方和, version2,todo:to be improved,需要保证每一项随dist增加都是单调不减的
        rc_err = dists_rc_normed
        neg_weight = self.neg_weight_fm_nrc_dist(dists_rc)
        rot_err = dists_rot_normed + (1-dists_rot_normed)*neg_weight
        rot_term = torch.clamp(rot_err, max=1.0) # min(1.,scale_err)
        if dists_scale is not None:
            scale_err = dists_scale_normed + (1-dists_scale_normed)*neg_weight
            scale_term = torch.clamp(scale_err, max=1.0) # min(1.,scale_err)
            dist_total_sq = self.w_rc * rc_err ** 2 + self.w_r * rot_term ** 2 + self.w_s * scale_term ** 2
        else:
            dist_total_sq = self.w_rc * rc_err**2 + self.w_r * rot_term**2

        #  取平方根，得到最终的距离,这使得 dist_true 的“单位”与 dist_pred 保持一致，损失函数更稳定
        dist_label = torch.sqrt(dist_total_sq) + 1e-7
        return dist_label

