import warnings, pdb
import argparse
from mmcv.utils import Config, Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
SEGMENTORS = Registry('segmentor')


def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):

    
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    out = build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
    pdb.set_trace()
    return out



cfg = dict(
            backbone=dict(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False
            ),
            decode_head=dict(
                in_channels=[96, 192, 384, 768],
                num_classes=150
            ),
            auxiliary_head=dict(
                in_channels=384,
                num_classes=150
            ))
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()
    return args

args = parse_args()
pdb.set_trace()

cfg = Config.fromfile(args.config)

build_segmentor(cfg, train_cfg=None, test_cfg=None)