import pandas as pd

from datasets import Dataset, DatasetDict, Value

from src.data.kaggle import download_dataset, unzip_file


def load_toxicity_dataset(folder, cache_folder, tokenizer, seed, train_size=0.9):
    cache_folder.mkdir(parents=True, exist_ok=True)

    cache_path = cache_folder / "jigsaw-unintended-bias-in-toxicity-classification"

    if cache_path.exists():
        return DatasetDict.load_from_disk(cache_path)

    zip_path = download_dataset(
        "jigsaw-unintended-bias-in-toxicity-classification", folder
    )

    ds_path = unzip_file(zip_path)

    df = pd.read_csv(ds_path / "train.csv")
    df = df.dropna(subset=["comment_text"])

    ds = Dataset.from_pandas(df, split="train")

    ds = ds.select_columns(["comment_text", "target"])
    ds = ds.rename_columns({"comment_text": "text", "target": "label"})

    ds = ds.shuffle(seed)
    ds = ds.train_test_split(train_size=train_size)

    ds = ds.map(preprocess(tokenizer))
    ds = ds.cast_column("label", Value("int32"))

    ds.save_to_disk(cache_path)

    return ds


def preprocess(tokenizer):

    def currying(data):
        tokenized = tokenizer(data["text"].lower(), truncation=True)

        tokenized["label"] = 1 if data["label"] > 0.5 else 0

        return tokenized

    return currying
