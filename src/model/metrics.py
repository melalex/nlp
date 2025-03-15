import evaluate
import numpy as np

metric = evaluate.combine(["precision", "recall", "f1"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return metric.compute(predictions=predictions, references=labels, average="macro")
