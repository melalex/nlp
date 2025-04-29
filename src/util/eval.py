import evaluate
import numpy as np
import tqdm


def evaluate_summery(summarizer, ds, max_target_length, model_type):
    bert = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    for sample in tqdm.tqdm(ds["test"]):
        pred = summarizer(sample["text"], max_length=max_target_length)[0]["summary_text"]
        predictions.append(pred)
        references.append(sample["summary"])

    rouge_score = rouge.compute(predictions=predictions, references=references)
    bert_score = bert.compute(
        predictions=predictions, references=references, model_type=model_type
    )
    del bert_score["hashcode"]
    bert_score = {key: np.average(values) for key, values in bert_score.items()}

    return rouge_score | bert_score
