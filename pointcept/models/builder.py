"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    assert cfg is not None and isinstance(cfg, dict), "cfg 必须是非空字典"
    """Build models."""
    return MODELS.build(cfg)
