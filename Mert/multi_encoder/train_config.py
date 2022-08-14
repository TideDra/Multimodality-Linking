from dataclasses import dataclass


@dataclass
class MultiEncoderTrainConfig:
    epochs = 1
    learning_rate = 5e-5
    batch_size = 20
    ckpt_name = "multi-encoder-flickr"
    ckpt_path = "/home/model_ckpt"
    max_ckpt_num = 5
