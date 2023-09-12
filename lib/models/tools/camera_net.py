import torch
import torch.nn as nn
import numpy as np
from lib.utils.transform import aa_to_rotmat, rot6d_to_rotmat, rot6d_to_aa
from lib.models.tools.geometry import *


# This code is borrow from https://github.com/ActiveVisionLab/nerfmm.git
class Focal_Net(nn.Module):
    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
        super(Focal_Net, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad
                )  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(
                        np.sqrt(init_focal / float(W)), requires_grad=False
                    ).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print("Focal init order need to be 1 or 2. Exit")
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad
                )  # (1, )
                self.fy = nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad
                )  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    if init_focal.shape[0] == 2:
                        coe_x = torch.tensor(
                            np.sqrt(init_focal[0] / float(W)), requires_grad=False
                        ).float()
                        coe_y = torch.tensor(
                            np.sqrt(init_focal[1] / float(H)), requires_grad=False
                        ).float()
                    else:
                        coe_x = torch.tensor(
                            np.sqrt(init_focal / float(W)), requires_grad=False
                        ).float()
                        coe_y = torch.tensor(
                            np.sqrt(init_focal / float(H)), requires_grad=False
                        ).float()

                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                    coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
                else:
                    print("Focal init order need to be 1 or 2. Exit")
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):  # the i=None is just to enable multi-gpu training
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fx**2 * self.W])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        return fxfy


# This code is borrow from https://github.com/ActiveVisionLab/nerfmm.git and https://github.com/quan-meng/gnerf.git
class Pose_Net(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, pose_mode="3d", init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(Pose_Net, self).__init__()
        self.num_cams = num_cams
        self.num_groups = 9 # iqscan dataset
        self.init_c2w = None
        self.pose_mode = pose_mode
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        if self.pose_mode == "3d":
            self.r = nn.Parameter(
                torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R
            )  # (N, 3)
        elif self.pose_mode == "6d":
            self.r = nn.Parameter(
                torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).repeat(num_cams, 1),
                requires_grad=learn_R,
            )  # (N, 6)
        else:
            raise ValueError(f"pose mode must be one of 3d or 6d, but got {self.pose_mode}")
        self.t = nn.Parameter(
            torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t
        )  # (N, 3)

    def forward(self, cam_ids):
        r = self.r[cam_ids]  # (N, 3) axis-angle
        t = self.t[cam_ids]  # (N, 3) or (N, 6)
        if self.pose_mode == "3d":
            R = aa_to_rotmat(r)  # (N, 3, 3)
        elif self.pose_mode == "6d":
            R = rot6d_to_rotmat(r)  # (N, 3)
        c2w = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (N, 3, 4)
        c2w = convert3x4_4x4(c2w)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_ids]

        return c2w
    
    def _l2_reg_loss(self, cam_ids):
        r = self.r[cam_ids] # (N, 3) axis-angle or (N, 6) 
        t = self.t[cam_ids] # (N, 3)
        
        if r.requires_grad:
            if self.pose_mode == "6d":
                r = rot6d_to_aa(r)
            reg_r = torch.mean(torch.linalg.norm(r, dim=-1), dim=0)
        else:
            reg_r = torch.tensor([0.], device=r.device, requires_grad=False)
            
        if t.requires_grad:
            reg_t = torch.mean(torch.linalg.norm(t, dim=-1), 0)
        else:
            reg_t = torch.tensor([0.], device=t.device, requires_grad=False)
        
        return reg_r + reg_t     
    
    
    def geometric_loss(self):
        cam_ids = torch.arange(self.num_cams)
        all_poses = self.forward(cam_ids)
        pose_groups = torch.chunk(all_poses, self.num_groups, dim=0)
        
        plane_reg_loss = []
        circle_reg_loss = []
        view_center_loss = []
        up_align_loss = []
        
        for i, pose_group in enumerate(pose_groups):
            cam_locs = pose_group[:, :3, 3].detach().cpu().numpy()
            
            C, n, r, d = fit_circle_3d(cam_locs)
            C = torch.as_tensor(C, device=all_poses.device)
            n = torch.as_tensor(n, device=all_poses.device)
            r = torch.as_tensor(r, device=all_poses.device)
            d = torch.as_tensor(d, device=all_poses.device)
            # plane regularization
            plane_reg_loss.append(self._plane_reg_loss(pose_group, n, d))
            # circle reg loss
            circle_reg_loss.append(self._circle_reg_loss(pose_group, C, r))
            # view center loss
            view_center_loss.append(self._view_center_loss(pose_group))
            # up vector alignment
            up_align_loss.append(self._up_vector_align_loss(pose_group, n))
            
        plane_reg_loss = torch.mean(torch.stack(plane_reg_loss))
        circle_reg_loss = torch.mean(torch.stack(circle_reg_loss))
        view_center_loss = torch.mean(torch.stack(view_center_loss))
        up_align_loss = torch.mean(torch.stack(up_align_loss))
        
        return plane_reg_loss + circle_reg_loss + view_center_loss + up_align_loss
    
    def _up_vector_align_loss(self, poses, normal):
        
        if normal.ndim == 1:
            normal = normal[None]
        
        right_vec, up_vec, view_vec = poses[:, :3, 0], -poses[:, :3, 1], poses[:, :3, 2]
        
        up_loss = torch.mean(torch.abs(torch.sum(up_vec * torch.cross(view_vec, normal), dim=1)))
        
        return up_loss
        
        
    def _plane_reg_loss(self, poses, normal, d):
        # plane regularization
        cam_locs = poses[:, :3, 3]
        p_distance = point_plane_distance(cam_locs, normal, d)

        plane_reg_loss = p_distance.mean()
        
        return plane_reg_loss
    
    
    def _circle_reg_loss(self, poses, C, r):
        
        cam_locs = poses[:, :3, 3]
        circle_reg_loss = torch.mean(torch.sum((cam_locs - C[None]) * (cam_locs - C[None]), dim=1), dim=0)
        
        return circle_reg_loss - r*r
    
    def _view_center_loss(self, poses):
        
        rays_o = poses[:, :3, 3]
        rays_d = poses[:, :3, 2]
        
        view_center, radius = center_radius_from_poses(poses)
        
        view_center_distance = point_line_distance(view_center, rays_o, rays_d)
        
        view_center_loss = torch.mean(view_center_distance, dim=0)
        
        return view_center_loss
    
     
    
    def cal_loss(self, cam_ids):
        
        # delta pose regularization
        reg_loss = self._l2_reg_loss(cam_ids)
        geo_loss = self.geometric_loss()
        
        return reg_loss + geo_loss


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat(
                [input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0
            )  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0
            )  # (4, 4)
            output[3, 3] = 1.0
    return output
