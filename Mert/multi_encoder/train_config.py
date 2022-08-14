from dataclasses import dataclass


@dataclass
class MultiEncoderTrainConfig:
    epochs = 1
    learning_rate = 5e-5
    ckpt_name = "multi-encoder-flickr"
    ckpt_path = "/home/model_ckpt"
    state_path = "/home/model_state"
    save_state_interval = 100
    max_ckpt_num = 5
    load_ckpt = True
