from pydantic import BaseModel

import pgx


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"