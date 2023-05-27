import logging
import os
import sys

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
import torch
import accelerate
from tqdm import tqdm
from torch.utils._pytree import tree_map
import torch.nn.functional as F
from skimage import measure
import trimesh
import pymeshlab as pml

configs.define_common_flags()


@torch.no_grad()
def evaluate_density(model, accelerator: accelerate.Accelerator,
                     points, config: configs.Config, std_value=0.0):
    """
    Evaluate a signed distance function (SDF) for a batch of points.

    Args:
        sdf: A callable function that takes a tensor of size (N, 3) containing
            3D points and returns a tensor of size (N,) with the SDF values.
        points: A torch tensor containing 3D points.

    Returns:
        A torch tensor with the SDF values evaluated at the given points.
    """
    z = []
    for _, pnts in enumerate(tqdm(torch.split(points, config.render_chunk_size, dim=0),
                                  desc="Evaluating density", leave=False,
                                  disable=not accelerator.is_main_process)):
        rays_remaining = pnts.shape[0] % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            pnts = torch.cat([pnts, torch.zeros_like(pnts[-padding:])], dim=0)
        else:
            padding = 0
        rays_per_host = pnts.shape[0] // accelerator.num_processes
        start, stop = accelerator.process_index * rays_per_host, \
                      (accelerator.process_index + 1) * rays_per_host
        chunk_means = pnts[start:stop]
        chunk_stds = torch.full_like(chunk_means[..., 0], std_value)
        raw_density = model.nerf_mlp.predict_density(chunk_means[:, None], chunk_stds[:, None], no_warp=True)[0]
        density = F.softplus(raw_density + model.nerf_mlp.density_bias)
        density = accelerator.gather(density)
        if padding > 0:
            density = density[: -padding]
        z.append(density)
    z = torch.cat(z, dim=0)
    return z


@torch.no_grad()
def evaluate_color(model, accelerator: accelerate.Accelerator,
                   points, config: configs.Config, std_value=0.0):
    """
    Evaluate a signed distance function (SDF) for a batch of points.

    Args:
        sdf: A callable function that takes a tensor of size (N, 3) containing
            3D points and returns a tensor of size (N,) with the SDF values.
        points: A torch tensor containing 3D points.

    Returns:
        A torch tensor with the SDF values evaluated at the given points.
    """
    z = []
    for _, pnts in enumerate(tqdm(torch.split(points, config.render_chunk_size, dim=0),
                                  desc="Evaluating color",
                                  disable=not accelerator.is_main_process)):
        rays_remaining = pnts.shape[0] % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            pnts = torch.cat([pnts, torch.zeros_like(pnts[-padding:])], dim=0)
        else:
            padding = 0
        rays_per_host = pnts.shape[0] // accelerator.num_processes
        start, stop = accelerator.process_index * rays_per_host, \
                      (accelerator.process_index + 1) * rays_per_host
        chunk_means = pnts[start:stop]
        chunk_stds = torch.full_like(chunk_means[..., 0], std_value)
        chunk_viewdirs = torch.tensor([0, 0, 1], dtype=torch.float32, device=chunk_means.device)
        chunk_viewdirs = torch.broadcast_to(chunk_viewdirs, chunk_means.shape)
        ray_results = model.nerf_mlp(False, chunk_means[:, None, None], chunk_stds[:, None, None],
                                     chunk_viewdirs)
        rgb = ray_results['rgb'][:, 0]
        rgb = accelerator.gather(rgb)
        if padding > 0:
            rgb = rgb[: -padding]
        z.append(rgb)
    z = torch.cat(z, dim=0)
    return z


@torch.no_grad()
def evaluate_color_projection(model, accelerator: accelerate.Accelerator, vertices, faces, config: configs.Config):
    normals = auto_normals(vertices, faces.long())
    viewdirs = -normals
    origins = vertices - 0.005 * viewdirs
    vc = []
    chunk = config.render_chunk_size
    model.num_levels = 1
    model.opaque_background = True
    for i in tqdm(range(0, origins.shape[0], chunk),
                  desc="Evaluating color projection",
                  disable=not accelerator.is_main_process):
        cur_chunk = min(chunk, origins.shape[0] - i)
        rays_remaining = cur_chunk % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            # raise
        else:
            padding = 0
        rays_per_host = (cur_chunk + accelerator.num_processes - 1) // accelerator.num_processes
        start = i + accelerator.process_index * rays_per_host
        stop = start + rays_per_host

        batch = {
            'origins': origins[start:stop],
            'directions': viewdirs[start:stop],
            'viewdirs': viewdirs[start:stop],
            'cam_dirs': viewdirs[start:stop],
            'radii': torch.full_like(origins[start:stop, ..., :1], 0.000723),
            'near': torch.full_like(origins[start:stop, ..., :1], 0),
            'far': torch.full_like(origins[start:stop, ..., :1], 0.01),
        }
        batch = accelerator.pad_across_processes(batch)
        with accelerator.autocast():
            renderings, ray_history = model(
                False,
                batch,
                compute_extras=False,
                train_frac=1)
        rgb = renderings[-1]['rgb']
        acc = renderings[-1]['acc']

        rgb /= acc.clamp_min(1e-5)[..., None]
        rgb = rgb.clamp(0, 1)

        rgb = accelerator.gather(rgb)
        rgb[torch.isnan(rgb) | torch.isinf(rgb)] = 1
        if padding > 0:
            rgb = rgb[: -padding]
        vc.append(rgb)
    vc = torch.cat(vc, dim=0)
    return vc


def auto_normals(verts, faces):
    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    v0 = verts[i0, :]
    v1 = verts[i1, :]
    v2 = verts[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(verts)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where((v_nrm ** 2).sum(dim=-1, keepdims=True) > 1e-20, v_nrm,
                        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=verts.device))
    v_nrm = F.normalize(v_nrm, dim=-1)
    return v_nrm


def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True, remesh_size=0.01, logger=None, main_process=True):
    # verts: [N, 3]
    # faces: [N, 3]
    tbar = tqdm(total=9, desc='Clean mesh', leave=False, disable=not main_process)
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')  # will copy!

    # filters
    tbar.set_description('Remove unreferenced vertices')
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces
    tbar.update()

    if v_pct > 0:
        tbar.set_description('Remove unreferenced vertices')
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct))  # 1/10000 of bounding box diagonal
    tbar.update()

    tbar.set_description('Remove duplicate faces')
    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    tbar.update()

    tbar.set_description('Remove null faces')
    ms.meshing_remove_null_faces()  # faces with area == 0
    tbar.update()

    if min_d > 0:
        tbar.set_description('Remove connected component by diameter')
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(min_d))
    tbar.update()

    if min_f > 0:
        tbar.set_description('Remove connected component by face number')
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)
    tbar.update()

    if repair:
        # tbar.set_description('Remove t vertices')
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        tbar.set_description('Repair non manifold edges')
        ms.meshing_repair_non_manifold_edges(method=0)
        tbar.update()
        tbar.set_description('Repair non manifold vertices')
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        tbar.update()
    else:
        tbar.update(2)
    if remesh:
        # tbar.set_description('Coord taubin smoothing')
        # ms.apply_coord_taubin_smoothing()
        tbar.set_description('Isotropic explicit remeshing')
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(remesh_size))
    tbar.update()

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    if logger is not None:
        logger.info(f'Mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True, logger=None):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh')  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    if logger is not None:
        logger.info(f'Mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


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

    visibility_path = os.path.join(config.mesh_path, 'visibility_mask_{:.1f}.pt'.format(config.mesh_radius))
    visibility_resolution = config.visibility_resolution
    if not os.path.exists(visibility_path):
        logger.info('Generate visibility mask...')
        # load dataset
        dataset = datasets.load_dataset('train', config.data_dir, config)
        dataloader = torch.utils.data.DataLoader(np.arange(len(dataset)),
                                                 num_workers=4,
                                                 shuffle=True,
                                                 batch_size=1,
                                                 collate_fn=dataset.collate_fn,
                                                 persistent_workers=True,
                                                 )

        visibility_mask = torch.ones(
            (1, 1, visibility_resolution, visibility_resolution, visibility_resolution), requires_grad=True
        ).to(device)
        visibility_mask.retain_grad()
        tbar = tqdm(dataloader, desc='Generating visibility grid', disable=not accelerator.is_main_process)
        for index, batch in enumerate(tbar):
            batch = accelerate.utils.send_to_device(batch, accelerator.device)

            rendering = models.render_image(model, accelerator,
                                            batch, False, 1, config,
                                            verbose=False, return_weights=True)

            coords = rendering['coord'].reshape(-1, 3)
            weights = rendering['weights'].reshape(-1)

            valid_points = coords[weights > config.valid_weight_thresh]
            valid_points /= config.mesh_radius
            # update mask based on ray samples
            with torch.enable_grad():
                out = torch.nn.functional.grid_sample(visibility_mask,
                                                      valid_points[None, None, None],
                                                      align_corners=True)
                out.sum().backward()
            tbar.set_postfix({"visibility_mask": (visibility_mask.grad > 0.0001).float().mean().item()})
            # if index == 10:
            #     break
        visibility_mask = (visibility_mask.grad > 0.0001).float()
        if accelerator.is_main_process:
            torch.save(visibility_mask.detach().cpu(), visibility_path)
    else:
        logger.info('Load visibility mask from {}'.format(visibility_path))
        visibility_mask = torch.load(visibility_path, map_location=device)

    space = config.mesh_radius * 2 / (config.visibility_resolution - 1)

    logger.info("Extract mesh from visibility mask...")
    visibility_mask_np = visibility_mask[0, 0].permute(2, 1, 0).detach().cpu().numpy()
    verts, faces, normals, values = measure.marching_cubes(
        volume=-visibility_mask_np,
        level=-0.5,
        spacing=(space, space, space))
    verts -= config.mesh_radius
    if config.extract_visibility:
        meshexport = trimesh.Trimesh(verts, faces)
        meshexport.export(os.path.join(config.mesh_path, "visibility_mask_{}.ply".format(config.mesh_radius)), "ply")
    logger.info("Extract visibility mask done.")

    # Initialize variables
    crop_n = 512
    grid_min = verts.min(axis=0)
    grid_max = verts.max(axis=0)
    space = ((grid_max - grid_min).prod() / config.mesh_voxels) ** (1 / 3)
    world_size = ((grid_max - grid_min) / space).astype(np.int32)
    Nx, Ny, Nz = np.maximum(1, world_size // crop_n)
    crop_n_x, crop_n_y, crop_n_z = world_size // [Nx, Ny, Nz]
    xs = np.linspace(grid_min[0], grid_max[0], Nx + 1)
    ys = np.linspace(grid_min[1], grid_max[1], Ny + 1)
    zs = np.linspace(grid_min[2], grid_max[2], Nz + 1)
    # Initialize meshes list
    meshes = []

    # Iterate over the grid
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                logger.info(f"Process grid cell ({i + 1}/{Nx}, {j + 1}/{Ny}, {k + 1}/{Nz})...")
                # Calculate grid cell boundaries
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                # Create point grid
                x = np.linspace(x_min, x_max, crop_n_x)
                y = np.linspace(y_min, y_max, crop_n_y)
                z = np.linspace(z_min, z_max, crop_n_z)
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T,
                                      dtype=torch.float,
                                      device=device)
                # Construct point pyramids
                points_tmp = points.reshape(crop_n_x, crop_n_y, crop_n_z, 3)[None]
                points_tmp /= config.mesh_radius
                current_mask = torch.nn.functional.grid_sample(visibility_mask, points_tmp, align_corners=True)
                current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]

                pts_density = evaluate_density(module, accelerator, points,
                                               config, std_value=config.std_value)

                z = pts_density.detach().cpu().numpy()

                # Skip if no surface found
                valid_z = z.reshape(crop_n_x, crop_n_y, crop_n_z)[current_mask]
                if valid_z.shape[0] <= 0 or (
                        np.min(valid_z) > config.isosurface_threshold or np.max(
                    valid_z) < config.isosurface_threshold
                ):
                    continue

                if not (np.min(z) > config.isosurface_threshold or np.max(z) < config.isosurface_threshold):
                    # Extract mesh
                    logger.info('Extract mesh...')
                    z = z.astype(np.float32)
                    verts, faces, _, _ = measure.marching_cubes(
                        volume=-z.reshape(crop_n_x, crop_n_y, crop_n_z),
                        level=-config.isosurface_threshold,
                        spacing=(
                            (x_max - x_min) / (crop_n_x - 1),
                            (y_max - y_min) / (crop_n_y - 1),
                            (z_max - z_min) / (crop_n_z - 1),
                        ),
                        mask=current_mask,
                    )
                    verts = verts + np.array([x_min, y_min, z_min])

                    meshcrop = trimesh.Trimesh(verts, faces)
                    logger.info('Extract vertices: {}, faces: {}'.format(meshcrop.vertices.shape[0],
                                                                         meshcrop.faces.shape[0]))
                    meshes.append(meshcrop)
    # Save mesh
    logger.info('Concatenate mesh...')
    combined_mesh = trimesh.util.concatenate(meshes)

    # from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/renderer.py
    # clean
    logger.info('Clean mesh...')
    vertices = combined_mesh.vertices.astype(np.float32)
    faces = combined_mesh.faces.astype(np.int32)
    vertices, faces = clean_mesh(vertices, faces,
                                 remesh=False, remesh_size=0.01,
                                 logger=logger, main_process=accelerator.is_main_process)

    # decimation
    if config.decimate_target > 0 and faces.shape[0] > config.decimate_target:
        logger.info('Decimate mesh...')
        vertices, triangles = decimate_mesh(vertices, faces, config.decimate_target, logger=logger)

    v = torch.from_numpy(vertices).contiguous().float().to(device)
    v = coord.inv_contract(2 * v)
    vertices = v.detach().cpu().numpy()
    f = torch.from_numpy(faces).contiguous().int().to(device)

    if config.vertex_color:
        # batched inference to avoid OOM
        logger.info('Evaluate mesh vertex color...')
        if config.vertex_projection:
            rgbs = evaluate_color_projection(module, accelerator, v, f, config)
        else:
            rgbs = evaluate_color(module, accelerator, v,
                                  config, std_value=config.std_value)
        rgbs = (rgbs * 255).detach().cpu().numpy().astype(np.uint8)
        if accelerator.is_main_process:
            logger.info('Export mesh (vertex color)...')
            mesh = trimesh.Trimesh(vertices, faces,
                                   vertex_colors=rgbs,
                                   process=False)  # important, process=True leads to seg fault...
            mesh.export(os.path.join(config.mesh_path, 'mesh_{}.ply'.format(config.mesh_radius)))
        logger.info('Finish extracting mesh.')
        return

    def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
        logger.info('Export mesh (atlas)...')
        # v, f: torch Tensor
        device = v.device
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        # unwrap uvs
        import xatlas
        import nvdiffrast.torch as dr
        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        logger.info(f'Running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 4  # for faster unwrap...
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

        # render uv maps
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        if h <= 2048 and w <= 2048:
            glctx = dr.RasterizeCudaContext()
        else:
            glctx = dr.RasterizeGLContext()

        rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

        # masked query
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        if mask.any():
            xyzs = xyzs[mask]  # [M, 3]

            # batched inference to avoid OOM
            all_feats = evaluate_color(module, accelerator, xyzs,
                                       config, std_value=config.std_value)
            feats[mask] = all_feats

        feats = feats.view(h, w, -1)
        mask = mask.view(h, w)

        # quantize [0.0, 1.0] to [0, 255]
        feats = feats.cpu().numpy()
        feats = (feats * 255).astype(np.uint8)

        ### NN search as an antialiasing ...
        mask = mask.cpu().numpy()

        inpaint_region = binary_dilation(mask, iterations=3)
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=2)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

        feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

        # do ssaa after the NN search, in numpy
        if ssaa > 1:
            feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(config.mesh_path, f'{name}albedo.png'), feats)

        # save obj (v, vt, f /)
        obj_file = os.path.join(config.mesh_path, f'{name}mesh.obj')
        mtl_file = os.path.join(config.mesh_path, f'{name}mesh.mtl')

        logger.info(f'writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info(f'writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info(f'writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                fp.write(f'vt {v[0]} {1 - v[1]} \n')

            logger.info(f'writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    # could be extremely slow
    _export(v, f)

    logger.info('Finish extracting mesh.')


if __name__ == '__main__':
    with gin.config_scope('bake'):
        app.run(main)
