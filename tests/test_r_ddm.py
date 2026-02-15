from pathlib import Path

import torch

from agents.r_ddm import IBLRDDMDataset, RDDMConfig, RDDMModel, rddm_collate_sessions


def test_rddm_dataset_single_session():
    dataset = IBLRDDMDataset(Path("data/ibl/reference_single_session.ndjson"), max_sessions=1)
    sample = dataset[0]
    assert sample.session_id
    assert len(sample.stimulus) == len(sample.action)
    assert sample.rt_seconds.min() >= 0.0


def test_rddm_model_forward_shapes():
    dataset = IBLRDDMDataset(Path("data/ibl/reference_single_session.ndjson"), max_sessions=1)
    batch = rddm_collate_sessions([dataset[0]])
    config = RDDMConfig(max_sessions=1)
    model = RDDMModel(config)
    stim = batch["stimulus"]
    block_prior = batch["block_prior"]
    prev_action = batch["prev_action"]
    prev_left = torch.where(prev_action == 0, 1.0, 0.0)
    prev_right = torch.where(prev_action == 1, 1.0, 0.0)
    features = torch.stack(
        [
            stim,
            stim.abs(),
            block_prior * config.prior_feature_scale,
            (block_prior - 0.5) * config.prior_feature_scale,
            prev_left * config.history_feature_scale,
            prev_right * config.history_feature_scale,
            (batch["prev_reward"] - 0.5) * config.history_feature_scale,
            (batch["prev_correct"] - 0.5) * config.history_feature_scale,
        ],
        dim=-1,
    ).float()
    outputs = model(features, batch["lengths"])
    assert outputs.choice_prob.shape == batch["stimulus"].shape
    assert torch.all(outputs.choice_prob > 0.0)
