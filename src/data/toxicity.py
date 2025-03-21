import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict, Value

from src.data.kaggle import download_competition_dataset, unzip_file

TOXICITY_LABEL_TO_ID = {
    "non-toxic": 0,
    "toxic": 1,
}

TOXICITY_ID_TO_LABEL = {v: k for k, v in TOXICITY_LABEL_TO_ID.items()}


def load_toxicity_dataset(
    folder,
    cache_folder,
    tokenizer,
    seed,
    identity_columns,
    num_proc=None,
    max_length=512,
    train_size=0.9,
):
    cache_folder.mkdir(parents=True, exist_ok=True)

    cache_path = cache_folder / "jigsaw-unintended-bias-in-toxicity-classification"

    if cache_path.exists():
        return DatasetDict.load_from_disk(cache_path)

    zip_path = download_competition_dataset(
        "jigsaw-unintended-bias-in-toxicity-classification", folder
    )

    ds_path = unzip_file(zip_path)

    df = pd.read_csv(ds_path / "train.csv")
    df = df.dropna(subset=["comment_text"])

    df["target"] = np.where(df["target"] >= 0.5, True, False)

    for col in identity_columns:
        df[col] = np.where(df[col] >= 0.5, True, False)

    ds = Dataset.from_pandas(df, split="train")

    ds = ds.rename_columns({"comment_text": "text", "target": "label"})
    ds = ds.map(preprocess(tokenizer), num_proc=num_proc)
    ds = ds.cast_column("label", Value("int32"))
    ds = ds.filter(filter_long_examples(max_length))

    ds = ds.shuffle(seed)
    ds = ds.train_test_split(train_size=train_size)

    ds.save_to_disk(cache_path)

    return ds


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def preprocess(tokenizer):

    def currying(data):
        return tokenizer(data["text"].lower(), truncation=False)

    return currying


def filter_long_examples(max_length):

    def currying(example):
        return len(example["input_ids"]) <= max_length

    return currying
