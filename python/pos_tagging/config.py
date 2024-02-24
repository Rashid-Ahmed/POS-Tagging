import logging
from typing import Optional

from pydantic import BaseModel
import tensorflow as tf
from pos_tagging.data.pos_tags import POStype

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    epochs: int = 5
    lr: float = 5e-5
    batch_size_per_device: int = 4
    warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    weight_decay: float = 0
    mixed_precision: bool = False

    @property
    def tf_strategy(self) -> tf.distribute.Strategy:

        if self.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        gpus = tf.config.list_physical_devices("GPU")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        if len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()

        return strategy


class ModelConfig(BaseModel):
    model_name: str = "microsoft/deberta-v3-large"
    tokenizer_name: Optional[str] = "microsoft/deberta-v3-large"
    fast_tokenizer: bool = True


class DataConfig(BaseModel):
    dataset_name: str = "conll2003"
    label2id = {item.value: index for index, item in enumerate(list(POStype))}
    max_token_length: Optional[int] = 256
    pad_to_max_length: bool = False
    is_split_into_words: bool = True
    label_all_tokens: bool = False


class Config(BaseModel):
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
