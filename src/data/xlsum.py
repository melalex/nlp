from datasets import load_dataset


def load_xlsum(tokenizer, max_input_length, max_target_length, language):
    ds = load_dataset("csebuetnlp/xlsum", language)
    ds = ds.map(
        preprocess_function(tokenizer, max_input_length, max_target_length),
        batched=True,
    )

    return ds


def preprocess_function(tokenizer, max_input_length, max_target_length):
    def currying(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return currying
