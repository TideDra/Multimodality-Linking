from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MultiEncoderConfig:
    num_layers: int = 6
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 8
    dropout: float = 0.1
    batch_first: bool = True

    forward_link: bool = False
    '''forward返回链表'''
    sole_flava: bool = False
    '''只使用一个FlavaModel而非三合一。使用此选项时不能进行Flava多模态融合'''
    augment_text: bool = True
    '''使用多层级表示增强text'''
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    '''phrase_level使用的kernel大小'''
    passes: List[str] = field(default_factory=lambda: ["gf", "bn"])
    '''融合顺序; Options: gf(GlobalFusion), mm(FlavaMultimodal), bn(BottleneckFusion), none'''