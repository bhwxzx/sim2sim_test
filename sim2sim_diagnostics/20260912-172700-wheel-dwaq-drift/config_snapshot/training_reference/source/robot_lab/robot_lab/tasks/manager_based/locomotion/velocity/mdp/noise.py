"""Observation noise that preserves deliberately masked wheel positions."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg, gaussian_noise, uniform_noise

__all__ = [
    "UniformJointPositionNoiseCfg", "uniform_joint_position_noise",
    "GaussianJointPositionNoiseCfg", "gaussian_joint_position_noise",
]


def _wheel_mask(data: torch.Tensor, cfg) -> torch.Tensor:
    if data.shape[-1] != len(cfg.joint_names):
        raise ValueError("joint_names must match the joint-position observation width")
    wheel_names = set(cfg.wheel_joint_names)
    return torch.tensor(
        [name in wheel_names for name in cfg.joint_names],
        dtype=torch.bool,
        device=data.device,
    )


def uniform_joint_position_noise(data: torch.Tensor, cfg: UniformJointPositionNoiseCfg) -> torch.Tensor:
    """Apply uniform noise only to non-wheel columns in observation order."""
    return torch.where(_wheel_mask(data, cfg), data, uniform_noise(data, cfg))


def gaussian_joint_position_noise(data: torch.Tensor, cfg: GaussianJointPositionNoiseCfg) -> torch.Tensor:
    """Apply Gaussian noise only to non-wheel columns in observation order."""
    return torch.where(_wheel_mask(data, cfg), data, gaussian_noise(data, cfg))


@configclass
class UniformJointPositionNoiseCfg(UniformNoiseCfg):
    """Uniform noise with exact joint names, serializable through Hydra/YAML.

    Names must follow the position observation's column order. Wheel columns
    retain their input values; the observation function is responsible for zeroing.
    """

    func = uniform_joint_position_noise
    joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING


@configclass
class GaussianJointPositionNoiseCfg(GaussianNoiseCfg):
    """Gaussian counterpart with joint names in position observation order."""

    func = gaussian_joint_position_noise
    joint_names: list[str] = MISSING
    wheel_joint_names: list[str] = MISSING
