import pandas as pd

from datasets import Dataset, DatasetDict, Value

from src.data.kaggle import download_dataset, unzip_file


def load_toxicity_dataset(folder, cache_folder, tokenizer, seed, train_size=0.9):
    cache_folder.mkdir(parents=True, exist_ok=True)

    cache_path = cache_folder / "jigsaw-unintended-bias-in-toxicity-classification"

    if cache_path.exists():
        return DatasetDict.load_from_disk(cache_path)

    def map_ds_row(data):
        tokenized = tokenizer(data["text"], truncation=True)

        tokenized["label"] = [1 if it > 0.5 else 0 for it in data["label"]]

        return tokenized

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

    ds = ds.map(map_ds_row, batched=True)
    ds = ds.cast_column("label", Value("int32"))

    ds.save_to_disk(cache_path)

    return ds
