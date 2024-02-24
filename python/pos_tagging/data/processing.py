from pos_tagging.config import Config
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from pos_tagging.data.utils import tokenize_and_align_labels


def load_data(config: Config):
    dataset = load_dataset(config.data.dataset_name)
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    train_dataset = train_dataset.remove_columns("chunk_tags")
    train_dataset = train_dataset.remove_columns("ner_tags")

    validation_dataset = validation_dataset.remove_columns("chunk_tags")
    validation_dataset = validation_dataset.remove_columns("ner_tags")

    return train_dataset, validation_dataset


def dataset_to_tf(train_dataset, validation_dataset, tokenizer, config, model):
    tokenized_train = train_dataset.map(lambda ex: tokenize_and_align_labels(ex, config, tokenizer),
                                        batched=True, batch_size=config.training.batch_size_per_device)
    tokenized_validation = validation_dataset.map(
        lambda ex: tokenize_and_align_labels(ex, config, tokenizer),
        batched=True,
        batch_size=config.training.batch_size_per_device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.model.model_name, return_tensors="tf")

    tf_train_set = model.prepare_tf_dataset(
        tokenized_train,
        shuffle=True,
        batch_size=config.training.batch_size_per_device,
        collate_fn=data_collator)

    tf_test_set = model.prepare_tf_dataset(
        tokenized_validation,
        shuffle=False,
        batch_size=config.training.batch_size_per_device,
        collate_fn=data_collator
    )

    return tf_train_set, tf_test_set
