import logging
from pathlib import Path
import os
from pos_tagging.config import Config
from pos_tagging.data.processing import load_data, dataset_to_tf
from pos_tagging.training.initializers import initialize_optimizer, initialize_model, initialize_tokenizer

logger = logging.getLogger(__name__)


def train(output_path: Path, config: Config):
    train_dataset, validation_dataset = load_data(config)
    train_batches_per_epoch = len(train_dataset) // config.training.batch_size_per_device
    validation_batches_per_epoch = len(validation_dataset) // config.training.batch_size_per_device
    auto_config, tokenizer = initialize_tokenizer(config)
    with config.training.tf_strategy.scope():

        optimizer = initialize_optimizer(config, train_batches_per_epoch)
        model = initialize_model(config, auto_config, tokenizer)
        tf_train_dataset, tf_validation_dataset = dataset_to_tf(train_dataset, validation_dataset, tokenizer, config, model)
        model.compile(optimizer=optimizer)
        model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=int(config.training.epochs),
            steps_per_epoch=train_batches_per_epoch,
            validation_steps=validation_batches_per_epoch,
        )

    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
    tokenizer.save_vocabulary(os.path.join(output_path, "tokenizer"))
    model.save_pretrained(output_path)
    model.save(output_path)