import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import cv2

import matplotlib.pyplot as plt
import glob
import pickle
import pyrender
import trimesh
from shapely import geometry
from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput as ModelOutput
from scipy.spatial.transform.rotation import Rotation as RRR


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        # J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        # self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        # self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        # extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)        #Additional 9 joints #Check doc/J_regressor_extra.png
        # joints = torch.cat([smpl_output.joints, extra_joints], dim=1)               #[N, 24 + 21, 3]  + [N, 9, 3]
        # joints = joints[:, self.joint_map, :]
        joints = smpl_output.joints
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self,
                 vertices,
                 focal_length=5000,
                 img_res=(224, 224),
                 faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                                   viewport_height=img_res[1],
                                                   point_size=2.0)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        minx, miny, minz = vertices.min(axis=(0, 1))
        maxx, maxy, maxz = vertices.max(axis=(0, 1))
        minx = minx - 0.5
        maxx = maxx + 0.5
        minz = minz - 0.5
        maxz = maxz + 0.5

        floor = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz],
                                  [maxx, minz]])
        self.floor = trimesh.creation.extrude_polygon(floor, 1e-5)
        self.floor.visual.face_colors = [0, 0, 0, 0.2]
        self.floor.apply_transform(self.rot)
        self.floor_pose = np.array(
            [[1, 0, 0, 0], [0, np.cos(np.pi / 2), -np.sin(np.pi / 2), miny],
             [0, np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 0, 1]])

        c = -np.pi / 6
        self.camera_pose = [[1, 0, 0, (minx + maxx) / 2],
                            [0, np.cos(c), -np.sin(c), 1.5],
                            [
                                0,
                                np.sin(c),
                                np.cos(c),
                                max(4, minz + (1.5 - miny) * 2, (maxx - minx))
                            ], [0, 0, 0, 1]]

    def __call__(self, vertices, camera_translation):

        floor_render = pyrender.Mesh.from_trimesh(self.floor, smooth=False)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            alphaMode='OPAQUE',
            baseColorFactor=(0.658, 0.214, 0.0114, 0.2))
        mesh = trimesh.Trimesh(vertices, self.faces)
        mesh.apply_transform(self.rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0), znear=0.5)

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=350)
        spot_l = pyrender.SpotLight(color=np.ones(3),
                                    intensity=300.0,
                                    innerConeAngle=np.pi / 16,
                                    outerConeAngle=np.pi / 6)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=300.0)

        scene = pyrender.Scene(bg_color=(1., 1., 1., 0.8),
                               ambient_light=(0.4, 0.4, 0.4))
        scene.add(floor_render, pose=self.floor_pose)
        scene.add(mesh, 'mesh')

        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        scene.add(camera, pose=self.camera_pose)

        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        color, rend_depth = self.renderer.render(scene, flags=flags)

        return color


class SMPLRender():

    def __init__(self, SMPL_MODEL_DIR):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=1,
                         create_transl=False).to(self.device)

        self.pred_camera_t = []
        self.focal_length = 110

    def init_renderer(self, res, smpl_param, is_headroot=False):
        poses = smpl_param['pred_pose']
        pred_rotmats = []
        for pose in poses:
            if pose.size == 72:
                pose = pose.reshape(-1, 3)
                pose = RRR.from_rotvec(pose).as_matrix()
                pose = pose.reshape(1, 24, 3, 3)
            pred_rotmats.append(
                torch.from_numpy(pose.astype(np.float32)[None]).to(
                    self.device))

        pred_rotmat = torch.cat(pred_rotmats, dim=0)

        pred_betas = torch.from_numpy(smpl_param['pred_shape'].reshape(
            1, 10).astype(np.float32)).to(self.device)
        pred_camera_t = smpl_param['pred_root'].reshape(1,
                                                        3).astype(np.float32)

        smpl_output = self.smpl(betas=pred_betas,
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)

        self.vertices = smpl_output.vertices.detach().cpu().numpy()

        pred_camera_t = pred_camera_t[0]

        if is_headroot:
            pred_camera_t = pred_camera_t - smpl_output.joints[
                0, 12].detach().cpu().numpy()

        self.pred_camera_t.append(pred_camera_t)

        self.renderer = Renderer(vertices=self.vertices,
                                 focal_length=self.focal_length,
                                 img_res=(res[1], res[0]),
                                 faces=self.smpl.faces)

    def render(self, index):
        renderImg = self.renderer(self.vertices[index, ...],
                                  self.pred_camera_t)
        return renderImg
