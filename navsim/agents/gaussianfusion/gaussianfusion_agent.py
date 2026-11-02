from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.agents.gaussianfusion.gaussianfusion_model import GaussianFusionModel
from navsim.agents.gaussianfusion.gaussianfusion_features import GaussianFusionTargetBuilder
from navsim.agents.gaussianfusion.gaussianfusion_features import GaussianFusionFeatureBuilder
from navsim.agents.gaussianfusion.gaussianfusion_config import GaussianFusionConfig
from navsim.agents.gaussianfusion.encoder.utils import dict_to_gpu
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from navsim.agents.gaussianfusion.gaussianfusion_loss import _bev_occ_loss
from navsim.agents.gaussianfusion.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop("type")
    return getattr(obj, type)(**cfg, **kwargs)

class GaussianFusionAgent(AbstractAgent):    
    def __init__(
            self,
            config: GaussianFusionConfig,
            lr: float,
            checkpoint_path: str = None,
    ):
        """
        Initialize GaussianFusion Agent
        
        Args:
            config: Configuration object containing model hyperparameters
            lr: Learning rate for optimization
            checkpoint_path: Path to pre-trained model checkpoint
        """
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data = config.cache_data  # Whether to use cached data only

        if not cache_data:
            # Initialize the main GaussianFusion model
            self._gaussianfusion_model = GaussianFusionModel(config)
            self.initialize()

        if not cache_data:  # Only for training
            # Initialize loss functions
            self.bce_logit_loss = nn.BCEWithLogitsLoss()  # Binary cross-entropy for scoring
            # Enable Ray for parallel processing
            self.ray = True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=40, log_to_driver=False)
                self.worker_map = worker_map
            
            # Load NavSim metric cache
            from .score_module.compute_navsim_score import get_scores
            print(os.getenv("NAVSIM_EXP_ROOT"))
            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths
            self.get_scores = get_scores

    def name(self) -> str:
        """Return the agent's class name for identification."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Initialize the agent by loading pre-trained weights if available.
        This is called after the agent is instantiated but before inference.
        """
        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, weights_only=True)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"), weights_only=True)[
                    "state_dict"]
            # Load weights with key mapping for compatibility
            state_dict = {k.replace("agent._gaussianfusion_model", "_gaussianfusion_model"): v for k, v in state_dict.items()}
            
            # Compare loaded state_dict with model's current state_dict
            model_state_dict = self.state_dict()
            loaded_keys = set(state_dict.keys())
            model_keys = set(model_state_dict.keys())
            
            # Print differences
            print("=" * 60)
            print("STATE_DICT COMPARISON")
            print("=" * 60)
            
            # Keys only in loaded state_dict
            only_in_loaded = loaded_keys - model_keys
            if only_in_loaded:
                print(f"\nKeys only in LOADED state_dict ({len(only_in_loaded)} keys):")
                for key in sorted(only_in_loaded):
                    print(f"  - {key}")
            
            # Keys only in model state_dict
            only_in_model = model_keys - loaded_keys
            if only_in_model:
                print(f"\nKeys only in MODEL state_dict ({len(only_in_model)} keys):")
                for key in sorted(only_in_model):
                    print(f"  - {key}")
            
            # Keys with different shapes
            common_keys = loaded_keys & model_keys
            shape_mismatches = []
            for key in common_keys:
                if state_dict[key].shape != model_state_dict[key].shape:
                    shape_mismatches.append((key, state_dict[key].shape, model_state_dict[key].shape))
            
            if shape_mismatches:
                print(f"\nKeys with SHAPE mismatches ({len(shape_mismatches)} keys):")
                for key, loaded_shape, model_shape in shape_mismatches:
                    print(f"  - {key}: loaded{loaded_shape} vs model{model_shape}")
            
            print(f"\nSummary:")
            print(f"  Total loaded keys: {len(loaded_keys)}")
            print(f"  Total model keys: {len(model_keys)}")
            print(f"  Common keys: {len(common_keys)}")
            print(f"  Keys only in loaded: {len(only_in_loaded)}")
            print(f"  Keys only in model: {len(only_in_model)}")
            print(f"  Shape mismatches: {len(shape_mismatches)}")
            print("=" * 60)
            
            self.load_state_dict(state_dict, strict=False)

    def model_to_cuda(self):
        self._gaussianfusion_model.cuda()

    def get_sensor_config(self):
        """
        Define the sensor configuration required by the agent.
        
        Returns:
            SensorConfig: Configuration specifying which cameras and sensors to use
                - cam_f0: Front camera (index 3)
                - cam_l0, cam_r0, cam_b0: Left, right, back cameras (index 3)
                - Other cameras and lidar disabled
        """
        return SensorConfig(
            cam_f0=[3],     # Front camera at index 3
            cam_l0=[3],     # Left camera at index 3
            cam_l1=[],      # Left camera 1 disabled
            cam_l2=[],      # Left camera 2 disabled
            cam_r0=[3],     # Right camera at index 3
            cam_r1=[],      # Right camera 1 disabled
            cam_r2=[],      # Right camera 2 disabled
            cam_b0=[],     # Back camera at index 3
            lidar_pc=[3],    # LiDAR point cloud disabled
        )
    def get_target_builders(self):
        """
        Return target builders for creating training targets.
        
        Returns:
            List of target builders used to process ground truth data
        """
        return [GaussianFusionTargetBuilder(config=self._config)]

    def get_feature_builders(self):
        """
        Return feature builders for processing input features.
        
        Returns:
            List of feature builders used to process sensor inputs
        """
        return [GaussianFusionFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GaussianFusion model.

        Args:
            features: Dictionary of input features
                - Camera images: shape [B, N_cam, C, H, W]
                - Other sensor data as configured
        
        Returns:
            Dictionary containing:
                - proposals: Generated trajectory proposals [B, N_proposals, T, 3]
                - pdm_score: Proposal scoring logits [B, N_proposals]
                - Other model outputs
        """
        return self._gaussianfusion_model(features, targets)

    def compute_trajectory(self, scene):
       
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene.get_agent_input()))

        for builder in self.get_target_builders():
            targets.update(builder.compute_targets(scene))

        targets.pop("token")
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        targets = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        features = dict_to_gpu(features)
        targets = dict_to_gpu(targets)
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features, targets)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        # extract trajectory
        return Trajectory(poses)
    
    def compute_score(self, targets, proposals, test=True):
        """
        Compute safety and feasibility scores for trajectory proposals.
        
        Args:
            targets: Dictionary containing:
                - trajectory: Ground truth trajectory [B, T, 3]
                - token: Scene tokens [B]
                - town_name: Map names [B]
                - lidar2world: Transform matrices [B, 4, 4]
            proposals: Predicted trajectory proposals [B, N_proposals, T, 3]
            test: Whether in test mode (affects metrics computed)
        
        Returns:
            If test=True:
                - final_scores: Final safety scores [B, N_proposals]
                - best_scores: Best score per sample [B]
                - all_scores: All intermediate scores [B, N_proposals, T]
                - l2_2s: L2 distance for first 2 seconds [B]
                - target_scores: Ground truth scores [B]
            If test=False:
                - final_scores: Final safety scores [B, N_proposals]
                - best_scores: Best score per sample [B]
                - target_scores: All scores including intermediate [B, N_proposals, T]
                - key_agent_corners: Agent corner coordinates [B, N_agents, 4, 2]
                - key_agent_labels: Agent validity labels [B, N_agents]
                - all_ego_areas: Area compliance flags [B, N_proposals, T, 3]
        """
        # Select appropriate metric cache based on training/testing mode
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]  # [B, T, 3]
        proposals = proposals.detach()  # [B, N_proposals, T, 3]

        # NavSim scoring: simpler trajectory evaluation
        data_points = [
            {
                "token": metric_cache_paths[token],  # Metric cache token
                "poses": poses,                      # Trajectory poses [N_proposals, T, 3]
                "test": test                         # Test mode flag
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        # Compute scores using parallel processing or sequential
        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        # Extract target scores from results
        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)  # [B, N_proposals, T]
        final_scores = target_scores[:, :, -1]  # [B, N_proposals] - final time step scores
        best_scores = torch.amax(final_scores, dim=-1)  # [B] - best score per sample

        if test:
            return final_scores.mean(), best_scores.mean(), final_scores
        else:
            return final_scores, best_scores, target_scores

    def score_loss(self, pred_logit, target_scores):
        """
        Compute scoring loss for trajectory proposals and agent predictions.
        
        Args:
            pred_logit: Primary score predictions [B, N_proposals, T]
            pred_logit2: Secondary score predictions [B, N_proposals, T] (optional)
            agents_state: Agent state predictions [B, N_agents, state_dim] (optional)
            pred_area_logits: Area compliance predictions [B, N_proposals, T, 3] (optional)
            target_scores: Ground truth scores [B, N_proposals, T]
            gt_states: Ground truth agent states [B, N_agents, state_dim]
            gt_valid: Agent validity mask [B, N_agents]
            gt_ego_areas: Ground truth area compliance [B, N_proposals, T, 3]
        
        Returns:
            Tuple of losses:
                - sub_score_loss: Loss on intermediate scores
                - final_score_loss: Loss on final time step scores
                - pred_ce_loss: Agent existence classification loss
                - pred_l1_loss: Agent state regression loss
                - pred_area_loss: Area compliance classification loss
        """

        # Primary scoring loss on intermediate time steps
        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])
        
        # Primary scoring loss on final time step
        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])

        return sub_score_loss, final_score_loss

    def diversity_loss(self, proposals):
        """
        Compute diversity loss to encourage diverse trajectory proposals.
        
        Args:
            proposals: Trajectory proposals [B, N_proposals, T, 3]
        
        Returns:
            inter_loss: Negative minimum inter-proposal distance (encourages diversity)
                       Shape: scalar
        """
        # Compute pairwise L1 distances between all proposals
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)  # [B, N_proposals, N_proposals]
        
        # Add small value to avoid division by zero
        dist = dist + (dist == 0)  # [B, N_proposals, N_proposals]
        
        # Find minimum distance for each proposal (excluding self-distance)
        inter_loss = -dist.amin(1).amin(1).mean()  # Negative to encourage larger distances
        
        return inter_loss

    def gaussianfusion_loss(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config):
        """
        Compute the complete GaussianFusion loss including trajectory, scoring, and auxiliary losses.
        
        Args:
            targets: Dictionary of ground truth data:
                - trajectory: Ground truth trajectory [B, T, 3]
                - Other target data for scoring and auxiliary tasks
            pred: Dictionary of model predictions:
                - proposals: Final trajectory proposals [B, N_proposals, T, 3]
                - proposal_list: Intermediate proposals from each decoder layer
                - pdm_score: Proposal scoring logits [B, N_proposals]
                - Other prediction outputs
            config: Configuration object with loss weights
        
        Returns:
            loss_dict: Dictionary containing:
                - loss: Total weighted loss
                - Individual loss components for monitoring
                - Evaluation metrics (score, best_score)
        """
        proposals = pred.get("proposals", None)              # [B, N_proposals, T, 3] - final proposals
        proposal_list = pred.get("proposal_list", [])      # List of [B, N_proposals, T, 3] - intermediate proposals
        target_trajectory = targets.get("trajectory", None)      # [B, T, 3] - ground truth

        if proposals is not None:
            # Compute safety and feasibility scores
            final_scores, best_scores, target_scores = self.compute_score(
                targets, proposals, test=False)

        # Progressive trajectory loss through decoder layers
        trajectory_loss = 0
      
        for proposals_i in proposal_list:
            # Minimum distance loss between proposals and ground truth
            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1).mean()  # [B] -> scalar
            
            # Accumulate trajectory loss with decay for earlier layers
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss
            

        # Scoring loss (if available)
        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss = self.score_loss(
                    pred["pred_logit"], 
                    target_scores
                )
        else:
            sub_score_loss = final_score_loss = 0

        # BEV semantic segmentation loss (if available)
        if "bev_map_pred" in pred.keys() and pred["bev_map_pred"] is not None:
            # bev_semantic_loss = F.cross_entropy(pred["bev_map_pred"], targets["bev_map_gt"].long())
            bev_semantic_loss = _bev_occ_loss(pred["bev_map_pred"], pred["bev_map_gt"])
        else:
            bev_semantic_loss = 0

        # Total weighted loss
        loss = (
                config.trajectory_weight * trajectory_loss        # Trajectory matching loss
                + config.sub_score_weight * sub_score_loss        # Sub scoring loss
                + config.final_score_weight * final_score_loss   # Final scoring loss
                + config.bev_semantic_weight * bev_semantic_loss # BEV segmentation loss
        )

        # Compute evaluation metrics
        if proposals is not None:
            pdm_score = pred["pdm_score"].detach()  # [B, N_proposals] - detached for evaluation
            top_proposals = torch.argmax(pdm_score, dim=1)  # [B] - best proposal indices
            score = final_scores[np.arange(len(final_scores)), top_proposals].mean()  # Average score of top proposals
            best_score = best_scores.mean()  # Average best score across batch
        else:
            score = best_score = 0

        # Package loss components and metrics
        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            "bev_semantic_loss": bev_semantic_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            "score": score,
            "best_score": best_score
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        """
        Main loss computation interface called during training.
        
        Args:
            features: Input features dictionary (not used directly)
            targets: Ground truth targets dictionary
            pred: Model predictions dictionary
        
        Returns:
            Dictionary of losses and metrics for backpropagation and logging
        """
        return self.gaussianfusion_loss(targets, pred, self._config)

    def get_optimizers(self):
        return torch.optim.Adam(self._gaussianfusion_model.parameters(), lr=self._lr)

    # def get_optimizers(
    #     self,
    # ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
    #     """Inherited, see superclass."""
    #     return self.get_coslr_optimizers()

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(
            type=self._config.optimizer_type,
            lr=self._lr,
            weight_decay=self._config.weight_decay,
            paramwise_cfg=self._config.opt_paramwise_cfg,
        )
        scheduler_cfg = dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)

        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg["name"]]

            for k, v in self._gaussianfusion_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg["name"].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._gaussianfusion_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg["name"].items()):
                cfg = {}
                if "lr_mult" in pg_cfg:
                    cfg["lr"] = optimizer_cfg["lr"] * pg_cfg["lr_mult"]
                optimizer.add_param_group({"params": pg, **cfg})

        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=20,
            warmup_epochs=1,
        )

        if "interval" in scheduler_cfg:
            scheduler = {"scheduler": scheduler, "interval": scheduler_cfg["interval"]}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def get_training_callbacks(self):
        """
        Get PyTorch Lightning callbacks for training.
        
        Returns:
            List of callbacks including model checkpointing based on validation score
        """
        checkpoint_cb = ModelCheckpoint(
            save_top_k=20,              # Keep top 20 checkpoints
            monitor='val/score_epoch',  # Monitor validation score
            filename='{epoch}-{step}',  # Checkpoint filename format
            mode="max"                  # Save checkpoints with maximum score
        )
        return [checkpoint_cb]