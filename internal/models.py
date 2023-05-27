import accelerate
import gin
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from tqdm import tqdm
from gridencoder import GridEncoder
from torch_scatter import segment_coo

gin.config.external_configurable(math.safe_exp, module='math')


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = None  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings)
        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            self.prop_mlp = PropMLP()
        else:
            for i in range(self.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i]))
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)

    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=False,
    ):
        """The mip-NeRF Model.

    Args:
      rand: random number generator (or None for deterministic output).
      batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        device = batch['origins'].device
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'], self.power_lambda)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                  self.near_anneal_init)
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                rand,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far))

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            means, stds = render.cast_rays(
                tdist,
                batch['origins'],
                batch['directions'],
                batch['cam_dirs'],
                batch['radii'],
                rand,
                std_scale=self.std_scale)

            # Push our Gaussians through one of our two MLPs.
            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            ray_results = mlp(
                rand,
                means, stds,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
            )

            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                batch['directions'],
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            # RawNeRF exposure logic.
            if batch.get('exposure_idx') is not None:
                # Scale output colors by the exposure.
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]

            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                batch['far'],
                compute_extras,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]

            if self.training:
                # Compute the hash decay loss for this level.
                idx = mlp.encoder.idx
                param = mlp.encoder.embeddings
                loss_hash_decay = segment_coo(param ** 2,
                                              idx,
                                              torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                              reduce='mean'
                                              ).mean()
                ray_results['loss_hash_decay'] = loss_hash_decay

            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        self.grid_num_levels = int(
            np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
        self.encoder = GridEncoder(input_dim=3,
                                   num_levels=self.grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   desired_resolution=self.grid_disired_resolution,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False)
        last_dim = self.encoder.output_dim
        if self.scale_featurization:
            last_dim += self.encoder.num_levels
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,
                                                     1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f"lin_glo_{self.net_depth_glo - 1}",
                                     nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, stds, rand=False, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
        weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
        features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
                                      self.encoder.idx,
                                      torch.zeros(self.grid_num_levels, device=weights.device),
                                      self.grid_num_levels,
                                      reduce='mean'
                                      )
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
            features = torch.cat([features, featurized_w], dim=-1)
        x = self.density_layer(features)
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        return raw_density, x, means.mean(dim=-2)

    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        if self.disable_density_normals:
            raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract = self.predict_density(means, stds, rand=rand, no_warp=no_warp)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


@torch.no_grad()
def render_image(model,
                 accelerator: accelerate.Accelerator,
                 batch,
                 rand,
                 train_frac,
                 config,
                 verbose=True,
                 return_weights=False):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
    accelerator: used for DDP.
    batch: a `Rays` pytree, the rays to be rendered.
    rand: if random
    config: A Config class.

  Returns:
    rgb: rendered color image.
    disp: rendered disparity image.
    acc: rendered accumulated weights per pixel.
  """
    model.eval()

    height, width = batch['origins'].shape[:2]
    num_rays = height * width
    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    global_rank = accelerator.process_index
    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk", leave=False,
                 disable=not (accelerator.is_main_process and verbose))

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
        actual_chunk_size = chunk_batch['origins'].shape[0]
        rays_remaining = actual_chunk_size % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0), chunk_batch)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
        start, stop = global_rank * rays_per_host, (global_rank + 1) * rays_per_host
        chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

        with accelerator.autocast():
            chunk_renderings, ray_history = model(rand,
                                                  chunk_batch,
                                                  train_frac=train_frac,
                                                  compute_extras=True)

        gather = lambda v: accelerator.gather(v.contiguous())[:-padding] \
            if padding > 0 else accelerator.gather(v.contiguous())
        # Unshard the renderings.
        chunk_renderings = tree_map(gather, chunk_renderings)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        if return_weights:
            chunk_rendering['weights'] = gather(ray_history[-1]['weights'])
            chunk_rendering['coord'] = gather(ray_history[-1]['coord'])
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]
    model.train()
    return rendering
