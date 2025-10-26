import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib
# Use a non-interactive backend suitable for servers/headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no-op tqdm if the package isn't available
    def tqdm(iterable, **kwargs):
        return iterable
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_frame
from navsim.visualization.plots import plot_bev_with_agent, plot_camera_with_agent
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.agents.gaussianfusion import gaussianfusion_agent, gaussianfusion_config
from navsim.visualization.plots import plot_cameras_frame


SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "all_scenes"

hydra.initialize(
    config_path="../navsim/planning/script/config/common/train_test_split/scene_filter"
)
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

agent_cfg = gaussianfusion_config.GaussianFusionConfig()
agent = gaussianfusion_agent.GaussianFusionAgent(
    agent_cfg,
    1e-4,
    "/home/liush/GaussianFusion/pretrained/epoch=24-step=66475.ckpt",
)

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=agent.get_sensor_config(),
)

# agent.initialize()
agent.eval()
agent.model_to_cuda()

# Create output directory for saved figures
output_dir = Path("exp/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)
bev_dir = output_dir
camera_dir = output_dir
bev_dir.mkdir(parents=True, exist_ok=True)
camera_dir.mkdir(parents=True, exist_ok=True)

# Try to provide total for tqdm if available
try:
    total_tokens = len(scene_loader.tokens)  # may fail if not sized
except Exception:
    total_tokens = None

for token in tqdm(scene_loader.tokens, total=total_tokens, desc="Rendering scenes"):
    scene = scene_loader.get_scene_from_token(token)
    frame_idx = scene.scene_metadata.num_history_frames - 1  # current frame
    fig, ax = plot_bev_with_agent(scene, agent)
    # Save BEV visualization to disk instead of showing a window
    out_path = bev_dir / f"{SPLIT}_{token}_frame{frame_idx}_bev.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # Save camera visualization to disk
    fig_cam, ax_cam = plot_camera_with_agent(scene, agent)
    out_cam_path = camera_dir / f"{SPLIT}_{token}_frame{frame_idx}_camera.png"
    fig_cam.savefig(out_cam_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_cam)

# fig, ax = plot_cameras_frame(scene, frame_idx)
# plt.show()
