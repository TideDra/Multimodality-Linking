from dataclasses import dataclass
@dataclass
class Config:
    batch_size:int=8
    epochs:int=10
config=Config()