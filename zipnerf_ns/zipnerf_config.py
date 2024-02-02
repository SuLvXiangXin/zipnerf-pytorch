"""
Nerfstudio ZipNerf Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from zipnerf_ns.zipnerf_datamanager import (
    ZipNerfDataManagerConfig,
)
from zipnerf_ns.zipnerf_model import ZipNerfModelConfig
from zipnerf_ns.zipnerf_pipeline import (
    ZipNerfPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


zipnerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="zipnerf", 
        steps_per_eval_batch=1000,
        steps_per_eval_image=5000,
        steps_per_save=5000,
        max_num_iterations=25000,
        mixed_precision=True,
        log_gradients=False,
        pipeline=ZipNerfPipelineConfig(
            datamanager=ZipNerfDataManagerConfig(
                dataparser=ColmapDataParserConfig(downscale_factor=4,orientation_method="up",center_method="poses", colmap_path="sparse/0"),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
            ),
            model=ZipNerfModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                gin_file=["configs/360.gin"],
                proposal_weights_anneal_max_num_iters=1000,
            ),
        ),
        optimizers={
            "model": {
                "optimizer": AdamOptimizerConfig(lr=8e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=1000,lr_final=1e-3, max_steps=25000)
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="An unofficial pytorch implementation of 'Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields' https://arxiv.org/abs/2304.06706. ",
)
