from pandas import DataFrame
from pos_tagging.config import Config
from transformers import AutoTokenizer


def tokenize_and_align_labels(
        examples: DataFrame,
        config: Config,
        tokenizer: AutoTokenizer.from_pretrained,
):
    padding = "max_length" if config.data.pad_to_max_length else False
    splitted = True if config.data.is_split_into_words else False
    tokenized_inputs = tokenizer(
        list(examples["tokens"]),
        max_length=config.data.max_token_length,
        padding=padding,
        truncation=True,
        is_split_into_words=splitted,
    )

    labels = []
    label_all_tokens = True if config.data.label_all_tokens else False
    for idx, labels_list in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)
            elif labels_list[word_idx] == 0:
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_list[word_idx])
            else:
                label_ids.append(labels_list[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels

    return tokenized_inputs
