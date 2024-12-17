# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C3k,
    C3k2,
    RotationAwareMultiHeadAttention,
    C2PSA,
    ELAN1,
    AConv,
    ARM,
    FFA,
    FEM,
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    DAPPM,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    ContrastiveHead,
    BNContrastiveHead,
    ADown,
    SPPELAN,
    CBFuse,
    CBLinear,
    Silence,
    PSA,
    C2fCIB,
    SCDown,
    RepVGGDW
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    Add,
    splitConv
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

from .Transformer.block import C3TR
from .InceptionNeXt.block import InceptionDWConv2d, InceptionNeXtBlock
from .MLPMixer.block import MLPMixer
from .RFAConv import RFAConv


__all__ = (
    "C2PSA",
    "C3k",
    "C3k2"
    "RotationAwareMultiHeadAttention",
    "InceptionNeXtBlock",
    "MSFM",
    "RFAConv",
    "RepNCSPELAN5",
    "ELAN1",
    "AConv",
    "ARM",
    "FFA",
    "MLPMixer",
    "InceptionDWConv2d",
    "FEM",
    "C3TR",
    "Fusion",
    "C2f_CloAtt",
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "DAPPM",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "PSA",
    "C2fCIB",
    "SCDown",
    "RepVGGDW",
    "v10Detect",
    "Add",
    "splitConv"
)
