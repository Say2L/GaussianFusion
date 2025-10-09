from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

from torch.utils.data import DataLoader
from navsim.planning.training.dataset import MetricDataset
import torch
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

def dict_to_gpu(input):
    if not isinstance(input, dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        return input

    for k, v in input.items():
        input[k] = dict_to_gpu(v)
    return input


def get_traj_results(cfg: DictConfig) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.eval()
    agent.model_to_cuda()

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    # scene_filter.log_names = log_names
    # scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    test_data = MetricDataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        # cache_path=cfg.metric_cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    dataloader = DataLoader(
        test_data,
        **cfg.dataloader.params,
        drop_last=True,
        shuffle=False,
    )
    # dataloader = DataLoader(test_data, **cfg.dataloader.params, shuffle=False)

    # tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    traj_results = {}

    ##########################################################################

    total_inference_time = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        features, targets, tokens = batch
        features = dict_to_gpu(features)
        # Remove 'tokens' key from targets if present
        if isinstance(targets, dict) and 'token' in targets:
            del targets['token']
        targets = dict_to_gpu(targets)
        start_time = time.time()

        with torch.no_grad():
            predictions = agent.forward(features, targets)
            trajectories = predictions["trajectory"].cpu().numpy()

        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time

        # 更新进度条描述，显示平均推理延时
        avg_inference_time = total_inference_time / (batch_idx + 1)
        tqdm.write(
            f"Batch {batch_idx + 1}/{num_batches} - Inference Time: {inference_time:.4f}s - Avg Inference Time: {avg_inference_time:.4f}s"
        )

        result_dict = {
            token: Trajectory(trajectory)
            for token, trajectory in zip(tokens, trajectories)
        }
        traj_results.update(result_dict)

    return traj_results


def run_pdm_score(
    args: List[Dict[str, Union[List[str], DictConfig]]],
) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    tokens = [a["token"] for a in args]
    trajectories = [a["trajectory"] for a in args]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # tokens_to_evaluate = list(set(tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}

        metric_cache_path = metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        trajectory = trajectories[idx]

        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        score_row.update(asdict(pdm_result))
        # except Exception as e:
        #     logger.warning(f"----------- Agent failed for token {token}:")
        #     traceback.print_exc()
        #     score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(
        set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    )
    num_missing_metric_cache_tokens = len(
        set(scene_loader.tokens) - set(metric_cache_loader.tokens)
    )
    num_unused_metric_cache_tokens = len(
        set(metric_cache_loader.tokens) - set(scene_loader.tokens)
    )
    if num_missing_metric_cache_tokens > 0:
        logger.warning(
            f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens."
        )
    if num_unused_metric_cache_tokens > 0:
        logger.warning(
            f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens."
        )
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
    tokens_to_trajectories = get_traj_results(cfg)
    data_points = [
        {
            "cfg": cfg,
            "token": token,
            "trajectory": traj,
        }
        for token, traj in tokens_to_trajectories.items()
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(
        worker, run_pdm_score, data_points
    )
    # score_rows = run_pdm_score(data_points)
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()
