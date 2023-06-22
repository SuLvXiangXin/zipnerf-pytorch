import glob
import logging
import os
import sys
import time

import cv2
import numpy as np
from absl import app
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import utils
from internal import coord
from internal import checkpoints
from internal import configs
import torch
import accelerate
from tqdm import tqdm
from torch.utils._pytree import tree_map
import torch.nn.functional as F
from skimage import measure
import trimesh
import pymeshlab as pml
from torch import Tensor

configs.define_common_flags()


class TSDF:
    def __init__(self, config: configs.Config, accelerator: accelerate.Accelerator):
        self.config = config
        self.device = accelerator.device
        self.accelerator = accelerator
        self.origin = torch.tensor([-config.tsdf_radius] * 3, dtype=torch.float32, device=self.device)
        self.voxel_size = 2 * config.tsdf_radius / (config.tsdf_resolution - 1)
        self.resolution = config.tsdf_resolution
        # create the voxel coordinates
        dim = torch.arange(self.resolution)
        grid = torch.stack(torch.meshgrid(dim, dim, dim, indexing="ij"), dim=0).reshape(3, -1)
        period = int(grid.shape[1] / accelerator.num_processes + 0.5)
        grid = grid[:, period * accelerator.process_index: period * (accelerator.process_index + 1)]
        self.voxel_coords = self.origin.view(3, 1) + grid.to(self.device) * self.voxel_size

        N = self.voxel_coords.shape[1]
        # make voxel_coords homogeneous
        voxel_world_coords = coord.inv_contract(self.voxel_coords.permute(1, 0)).permute(1, 0).view(3, -1)
        # voxel_world_coords = self.voxel_coords.view(3, -1)
        voxel_world_coords = torch.cat(
            [voxel_world_coords, torch.ones(1, voxel_world_coords.shape[1], device=self.device)], dim=0
        )
        voxel_world_coords = voxel_world_coords.unsqueeze(0)  # [1, 4, N]
        self.voxel_world_coords = voxel_world_coords.expand(-1, *voxel_world_coords.shape[1:])  # [1, 4, N]

        # initialize the values and weights
        self.values = torch.ones(N, dtype=torch.float32,
                                 device=self.device)
        self.weights = torch.zeros(N, dtype=torch.float32,
                                   device=self.device)
        self.colors = torch.zeros(N, 3, dtype=torch.float32,
                                  device=self.device)

    @property
    def truncation(self):
        """Returns the truncation distance."""
        # TODO: clean this up
        truncation = self.voxel_size * self.config.truncation_margin
        return truncation

    def export_mesh(self, path):
        """Extracts a mesh using marching cubes."""
        # run marching cubes on CPU
        tsdf_values = self.values.clamp(-1, 1)
        mask = self.voxel_world_coords[:, :3].permute(0, 2, 1).norm(p=2, dim=-1) > self.config.tsdf_max_radius
        tsdf_values[mask.reshape(self.values.shape)] = 1.

        tsdf_values_np = self.accelerator.gather(tsdf_values).cpu().reshape((self.resolution, self.resolution, self.resolution)).numpy()
        color_values_np = self.accelerator.gather(self.colors).cpu().reshape((self.resolution, self.resolution, self.resolution, 3)).numpy()

        # # for OOM(resolution > 512)
        # tsdf_values_np = tsdf_values.cpu().numpy()
        # color_values_np = self.colors.cpu().numpy()
        # path_dir = os.path.dirname(path)
        # np.save(os.path.join(path_dir, 'tsdf_values_tmp_{}.npy'.format(self.accelerator.process_index)), tsdf_values_np)
        # np.save(os.path.join(path_dir, 'color_values_tmp_{}.npy'.format(self.accelerator.process_index)), color_values_np)
        # self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # print('Start marching cubes')
            # tsdf_values_np = np.concatenate([np.load(os.path.join(path_dir, 'tsdf_values_tmp_{}.npy'.format(i)), allow_pickle=True) for i in
            #                                  range(self.accelerator.num_processes)]).reshape((self.resolution, self.resolution, self.resolution))
            # color_values_np = np.concatenate([np.load(os.path.join(path_dir, 'color_values_tmp_{}.npy'.format(i)), allow_pickle=True) for i in
            #                                   range(self.accelerator.num_processes)]).reshape((self.resolution, self.resolution, self.resolution, 3))
            # print('After concatenate')
            # os.system('rm {}'.format(os.path.join(path_dir, 'tsdf_values_tmp_*.npy')))
            # os.system('rm {}'.format(os.path.join(path_dir, 'color_values_tmp_*.npy')))
            vertices, faces, normals, _ = measure.marching_cubes(
                tsdf_values_np,
                level=0,
                allow_degenerate=False,
            )

            vertices_indices = np.round(vertices).astype(int)
            colors = color_values_np[vertices_indices[:, 0], vertices_indices[:, 1], vertices_indices[:, 2]]

            # move vertices back to world space
            vertices = self.origin.cpu().numpy() + vertices * self.voxel_size
            vertices = coord.inv_contract_np(vertices)
            trimesh.Trimesh(vertices=vertices,
                            faces=faces,
                            normals=normals,
                            vertex_colors=colors,
                            ).export(path)

    @torch.no_grad()
    def integrate_tsdf(
            self,
            c2w,
            K,
            depth_images,
            color_images=None,
    ):
        """Integrates a batch of depth images into the TSDF.

        Args:
            c2w: The camera extrinsics.
            K: The camera intrinsics.
            depth_images: The depth images to integrate.
            color_images: The color images to integrate.
        """
        batch_size = c2w.shape[0]
        shape = self.voxel_coords.shape[1:]

        # Project voxel_coords into image space...
        image_size = torch.tensor(
            [depth_images.shape[-1], depth_images.shape[-2]], device=self.device
        )  # [width, height]

        # make voxel_coords homogeneous
        voxel_world_coords = self.voxel_world_coords.expand(batch_size,
                                                            *self.voxel_world_coords.shape[1:])  # [batch, 4, N]

        voxel_cam_coords = torch.bmm(torch.inverse(c2w), voxel_world_coords)  # [batch, 4, N]

        # flip the z axis
        voxel_cam_coords[:, 2, :] = -voxel_cam_coords[:, 2, :]
        # flip the y axis
        voxel_cam_coords[:, 1, :] = -voxel_cam_coords[:, 1, :]

        # # we need the distance of the point to the camera, not the z coordinate
        # # TODO: why is this not the z coordinate?
        # voxel_depth = torch.sqrt(torch.sum(voxel_cam_coords[:, :3, :] ** 2, dim=-2, keepdim=True))  # [batch, 1, N]

        voxel_cam_coords_z = voxel_cam_coords[:, 2:3, :]
        voxel_depth = voxel_cam_coords_z

        voxel_cam_points = torch.bmm(K[None].expand(batch_size, -1, -1),
                                     voxel_cam_coords[:, 0:3, :] / voxel_cam_coords_z)  # [batch, 3, N]
        voxel_pixel_coords = voxel_cam_points[:, :2, :]  # [batch, 2, N]

        # Sample the depth images with grid sample...

        grid = voxel_pixel_coords.permute(0, 2, 1)  # [batch, N, 2]
        # normalize grid to [-1, 1]
        grid = 2.0 * grid / image_size.view(1, 1, 2) - 1.0  # [batch, N, 2]
        grid = grid[:, None]  # [batch, 1, N, 2]
        # depth
        sampled_depth = F.grid_sample(
            input=depth_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
        )  # [batch, N, 1]
        sampled_depth = sampled_depth.squeeze(2)  # [batch, 1, N]
        # colors
        sampled_colors = None
        if color_images is not None:
            sampled_colors = F.grid_sample(
                input=color_images, grid=grid, mode="nearest", padding_mode="zeros", align_corners=False
            )  # [batch, N, 3]
            sampled_colors = sampled_colors.squeeze(2)  # [batch, 3, N]

        dist = sampled_depth - voxel_depth  # [batch, 1, N]

        # x = self.voxel_world_coords[:, :3].permute(0, 2, 1)
        # eps = torch.finfo(x.dtype).eps
        # x_mag_sq = torch.sum(x ** 2, dim=-1).clamp_min(eps)
        # truncation_weight = torch.where(x_mag_sq <= 1, torch.ones_like(x_mag_sq),
        #                                 ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq))
        # truncation = truncation_weight.reciprocal() * self.truncation

        truncation = self.truncation

        tsdf_values = torch.clamp(dist / truncation, min=-1.0, max=1.0)  # [batch, 1, N]
        valid_points = (voxel_depth > 0) & (sampled_depth > 0) & (dist > -self.truncation)  # [batch, 1, N]

        # Sequentially update the TSDF...
        for i in range(batch_size):
            valid_points_i = valid_points[i]
            valid_points_i_shape = valid_points_i.view(*shape)  # [xdim, ydim, zdim]

            # the old values
            old_tsdf_values_i = self.values[valid_points_i_shape]
            old_weights_i = self.weights[valid_points_i_shape]

            # the new values
            # TODO: let the new weight be configurable
            new_tsdf_values_i = tsdf_values[i][valid_points_i]
            new_weights_i = 1.0

            total_weights = old_weights_i + new_weights_i

            self.values[valid_points_i_shape] = (old_tsdf_values_i * old_weights_i +
                                                 new_tsdf_values_i * new_weights_i) / total_weights
            # self.weights[valid_points_i_shape] = torch.clamp(total_weights, max=1.0)
            self.weights[valid_points_i_shape] = total_weights

            if sampled_colors is not None:
                old_colors_i = self.colors[valid_points_i_shape]  # [M, 3]
                new_colors_i = sampled_colors[i][:, valid_points_i.squeeze(0)].permute(1, 0)  # [M, 3]
                self.colors[valid_points_i_shape] = (old_colors_i * old_weights_i[:, None] +
                                                     new_colors_i * new_weights_i) / total_weights[:, None]


def main(unused_argv):
    config = configs.load_config()
    config.compute_visibility = True

    config.exp_path = os.path.join("exp", config.exp_name)
    config.mesh_path = os.path.join("exp", config.exp_name, "mesh")
    config.checkpoint_dir = os.path.join(config.exp_path, 'checkpoints')
    os.makedirs(config.mesh_path, exist_ok=True)

    # accelerator for DDP
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # setup logger
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(config.exp_path, 'log_extract.txt'))],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    accelerate.utils.set_seed(config.seed, device_specific=True)

    # setup model and optimizer
    model = models.Model(config=config)
    model = accelerator.prepare(model)
    step = checkpoints.restore_checkpoint(config.checkpoint_dir, accelerator, logger)
    model.eval()
    module = accelerator.unwrap_model(model)

    dataset = datasets.load_dataset('train', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                             shuffle=False,
                                             batch_size=1,
                                             collate_fn=dataset.collate_fn,
                                             )
    dataiter = iter(dataloader)
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    out_name = f'train_preds_step_{step}'
    out_dir = os.path.join(config.mesh_path, out_name)
    utils.makedirs(out_dir)
    logger.info("Render trainset in {}".format(out_dir))

    path_fn = lambda x: os.path.join(out_dir, x)

    # Ensure sufficient zero-padding of image indices in output filenames.
    zpad = max(3, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    for idx in range(dataset.size):
        # If current image and next image both already exist, skip ahead.
        idx_str = idx_to_str(idx)
        curr_file = path_fn(f'color_{idx_str}.png')
        if utils.file_exists(curr_file):
            logger.info(f'Image {idx + 1}/{dataset.size} already exists, skipping')
            continue
        batch = next(dataiter)
        batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
        logger.info(f'Evaluating image {idx + 1}/{dataset.size}')
        eval_start_time = time.time()
        rendering = models.render_image(model, accelerator,
                                        batch, False, 1, config)

        logger.info(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

        if accelerator.is_main_process:  # Only record via host 0.
            rendering['rgb'] = postprocess_fn(rendering['rgb'])
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
            utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))
            utils.save_img_f32(rendering['distance_mean'],
                               path_fn(f'distance_mean_{idx_str}.tiff'))
            utils.save_img_f32(rendering['distance_median'],
                               path_fn(f'distance_median_{idx_str}.tiff'))

    # if accelerator.is_main_process:
    tsdf = TSDF(config, accelerator)

    c2w = torch.from_numpy(dataset.camtoworlds[:, :3, :4]).float().to(device)

    # make c2w homogeneous
    c2w = torch.cat([c2w, torch.zeros(c2w.shape[0], 1, 4, device=device)], dim=1)
    c2w[:, 3, 3] = 1
    K = torch.from_numpy(dataset.pixtocams).float().to(device).inverse()

    logger.info('Reading images')
    rgb_files = sorted(glob.glob(path_fn('color_*.png')))
    depth_files = sorted(glob.glob(path_fn('distance_median_*.tiff')))
    assert len(rgb_files) == len(depth_files)
    color_images = []
    depth_images = []
    for rgb_file, depth_file in zip(tqdm(rgb_files, disable=not accelerator.is_main_process), depth_files):
        color_images.append(utils.load_img(rgb_file) / 255)
        depth_images.append(utils.load_img(depth_file)[..., None])

    color_images = torch.tensor(np.array(color_images), device=device).permute(0, 3, 1, 2)  # shape (N, 3, H, W)
    depth_images = torch.tensor(np.array(depth_images), device=device).permute(0, 3, 1, 2)  # shape (N, 1, H, W)

    batch_size = 1
    logger.info("Integrating the TSDF")
    for i in tqdm(range(0, len(c2w), batch_size), disable=not accelerator.is_main_process):
        tsdf.integrate_tsdf(
            c2w[i: i + batch_size],
            K,
            depth_images[i: i + batch_size],
            color_images=color_images[i: i + batch_size],
        )

    logger.info("Saving TSDF Mesh")
    tsdf.export_mesh(os.path.join(config.mesh_path, "tsdf_mesh.ply"))
    accelerator.wait_for_everyone()
    logger.info('Finish extracting mesh using TSDF.')


if __name__ == '__main__':
    with gin.config_scope('bake'):
        app.run(main)
