
import os
os.environ['DISPLAY'] = ':0.0'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["MUJOCO_GL"] = "osmesa"
from argparse import ArgumentParser
import numpy as np
import OpenGL.GL as gl
import imageio
import cv2
import random
import torch
import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from mGPT.render.pyrender.smpl_render import SMPLRender


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--joints_path', type=str, help='Path to joints data')
    parser.add_argument('--method', type=str, help='Method for rendering')
    parser.add_argument('--output_mp4_path', type=str, help='Path to output MP4 file')
    parser.add_argument('--smpl_model_path', type=str, help='Path to SMPL model')

    args = parser.parse_args()

    joints_path = args.joints_path
    method = args.method
    output_mp4_path = args.output_mp4_path
    smpl_model_path = args.smpl_model_path


    data = np.load(joints_path)

    if method == 'slow':
        if len(data.shape) == 4:
            data = data[0]
        data = data - data[0, 0]
        pose_generator = HybrIKJointsToRotmat()
        pose = pose_generator(data)
        pose = np.concatenate([
            pose,
            np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
        ], 1)
        shape = [768, 768]
        render = SMPLRender(smpl_model_path)

        r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
        vid = []
        aroot = data[:, 0]
        aroot[:, 1:] = -aroot[:, 1:]
        params = dict(pred_shape=np.zeros([1, 10]),
                        pred_root=aroot,
                        pred_pose=pose)
        render.init_renderer([shape[0], shape[1], 3], params)
        for i in range(data.shape[0]):
            renderImg = render.render(i)
            vid.append(renderImg)

        out = np.stack(vid, axis=0)
        output_gif_path = output_mp4_path[:-4] + '.gif'
        imageio.mimwrite(output_gif_path, out, duration=50)
        out_video = mp.VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)

    elif method == 'fast':
        output_gif_path = output_mp4_path[:-4] + '.gif'
        if len(data.shape) == 3:
            data = data[None]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
        out_video = mp.VideoFileClip(output_gif_path)
        out_video.write_videofile(output_mp4_path)
