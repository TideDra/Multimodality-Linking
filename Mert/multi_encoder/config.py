from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MultiEncoderConfig:
    num_layers: int = 6
    d_bottleneck: int = 4
    hidden_size: int = 768
    nhead: int = 8
    dropout = 0.1
    batch_first = True

    forward_link = False
    '''forward返回链表'''
    sole_flava = False
    '''只使用一个FlavaModel而非三合一。使用此选项时不能进行Flava多模态融合'''
    augment_text = True
    '''使用多层级表示增强text'''
    conv_kernel_sizes = [3, 3, 3]
    '''phrase_level使用的kernel大小'''
    passes = ["gf", "bn"]
    '''融合顺序; Options: gf(GlobalFusion), mm(FlavaMultimodal), bn(BottleneckFusion), none'''