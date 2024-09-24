from dataclasses import dataclass, field
import importlib
import os
from typing import Dict, List, Literal, Type
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
import torch
from torch.nn import Parameter
from internal import train_utils
from internal.configs import Config
from internal.models import Model as zipnerf
import gin
import numpy as np
from nerfstudio.utils import colormaps
@dataclass
class ZipNerfModelConfig(ModelConfig):
    gin_file: list = None 
    """Config files list to load default setting of Model/NerfMLP/PropMLP as zipnerf-pytorch"""
    compute_extras: bool = True
    """if True, compute extra quantities besides color."""
    proposal_weights_anneal_max_num_iters: int = 1000  
    """Max num iterations for the annealing function. Set to the value of max_train_iterations to have same behavior as zipnerf-pytorch."""
    rand: bool = True
    """random number generator (or None for deterministic output)."""
    zero_glo: bool = False
    """if True, when using GLO pass in vector of zeros."""
    background_color: Literal["random", "black", "white"] = "white"
    """Whether to randomize the background color."""
    _target: Type = field(default_factory=lambda: ZipNerfModel)

class ZipNerfModel(Model):
    config: ZipNerfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # update default setting
        # gin.parse_config_files_and_bindings(self.config.gin_file, None)
        gin_files = []
        for g in self.config.gin_file:
            if os.path.exists(g):
                gin_files.append(g)
            else:
                package_path = importlib.util.find_spec("zipnerf_ns").origin.split('/')[:-2]
                package_path = '/'.join(package_path)
                gin_files.append(package_path+'/'+g)
        gin.parse_config_files_and_bindings(gin_files, None)
        config = Config()

        self.zipnerf = zipnerf(config=config)
        
        self.collider = NearFarCollider(near_plane=self.zipnerf.config.near, far_plane=self.zipnerf.config.far)
        self.step = 0

        # Renderer
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def construct_batch_from_raybundle(self, ray_bundle):
        batch = {}
        batch['origins'] = ray_bundle.origins
        batch['directions'] = ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        batch['viewdirs'] = ray_bundle.directions
        batch['radii'] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        batch['cam_idx'] = ray_bundle.camera_indices
        batch['near'] = ray_bundle.nears
        batch['far'] = ray_bundle.fars
        batch['cam_dirs'] = None  # did not be calculated in raybundle
        # batch['imageplane'] = None
        # batch['exposure_values'] = None
        return batch
    
    def get_outputs(self, ray_bundle: RayBundle):
        ray_bundle.metadata["viewdirs"] = ray_bundle.directions
        ray_bundle.metadata["radii"] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        ray_bundle.directions = ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        
        if self.training:
            anneal_frac = np.clip(self.step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)
        else:
            anneal_frac = 1.0
        batch = self.construct_batch_from_raybundle(ray_bundle)

        renderings, ray_history = self.zipnerf(
                rand=self.config.rand if self.training else False,  # set to false when evaluating or rendering
                batch=batch,
                train_frac=anneal_frac, 
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True) # set to True when evaluating or rendering
        
        outputs={}

        # showed by viewer
        outputs['rgb']=renderings[2]['rgb']
        outputs['depth']=renderings[2]['depth'].unsqueeze(-1)
        outputs['accumulation']=renderings[2]['acc']
        if self.config.compute_extras:
            outputs['distance_mean']=renderings[2]['distance_mean']
            outputs['distance_median']=renderings[2]['distance_median']

        # for loss calculation
        outputs['renderings']=renderings
        outputs['ray_history'] = ray_history
        return outputs
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def set_step(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
            """Returns the parameter groups needed to optimizer your model components."""
            param_groups = {}
            param_groups["model"] = list(self.parameters())
            return param_groups
    

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
        metrics_dict = {}
        gt_rgb = batch['image'].to(self.device)
        predicted_rgb = outputs['rgb']
        metrics_dict["psnr"] = self.psnr(gt_rgb, predicted_rgb)
        return metrics_dict
        
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        loss_dict={}
        batch['lossmult'] = torch.Tensor([1.]).to(self.device)
        
        data_loss, stats = train_utils.compute_data_loss(batch, outputs['renderings'], self.zipnerf.config)
        loss_dict['data'] = data_loss
        
        if self.training:
            # interlevel loss in MipNeRF360
            # if self.config.interlevel_loss_mult > 0 and not self.config.single_mlp:
            #     loss_dict['interlevel'] = train_utils.interlevel_loss(outputs['ray_history'], self.config)

            # interlevel loss in ZipNeRF360
            if self.zipnerf.config.anti_interlevel_loss_mult > 0 and not self.zipnerf.single_mlp:
                loss_dict['anti_interlevel'] = train_utils.anti_interlevel_loss(outputs['ray_history'], self.zipnerf.config)

            # distortion loss
            if self.zipnerf.config.distortion_loss_mult > 0:
                loss_dict['distortion'] = train_utils.distortion_loss(outputs['ray_history'], self.zipnerf.config)

            # opacity loss
            # if self.config.opacity_loss_mult > 0:
            #     loss_dict['opacity'] = train_utils.opacity_loss(outputs['rgb'], self.config)

            # # orientation loss in RefNeRF
            # if (self.config.orientation_coarse_loss_mult > 0 or
            #         self.config.orientation_loss_mult > 0):
            #     loss_dict['orientation'] = train_utils.orientation_loss(batch, self.config, outputs['ray_history'],
            #                                                             self.config)
            # hash grid l2 weight decay
            if self.zipnerf.config.hash_decay_mults > 0:
                loss_dict['hash_decay'] = train_utils.hash_decay_loss(outputs['ray_history'], self.zipnerf.config)

            # # normal supervision loss in RefNeRF
            # if (self.config.predicted_normal_coarse_loss_mult > 0 or
            #         self.config.predicted_normal_loss_mult > 0):
            #     loss_dict['predicted_normals'] = train_utils.predicted_normal_loss(
            #         self.config, outputs['ray_history'], self.config)
        return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]: # type: ignore
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        gt_rgb = batch["image"].to(self.device)

        predicted_rgb = outputs["rgb"]
        # print('min,max:',predicted_rgb.min(),predicted_rgb.max())
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(gt_rgb, predicted_rgb).item()),
            "ssim": float(self.ssim(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0))),
            "lpips": float(self.lpips(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0)))
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict