import evaluate
import logging
import numpy as np
from evaluate import load
logger = logging.getLogger(__name__)
 

class EvaluateTool(object):
    def __init__(self):
        pass

    # def evaluate(self, preds, golds):
    #     # Get number of gold labels
    #     num_samples = len(golds)
        
    #     # Create a list of "Yes" as reference labels
    #     golds = ["1"] * num_samples

    #     metric = evaluate.load("accuracy")
    #     return metric.compute(references=golds, predictions=preds)

    def evaluate(self, preds, refs):
        # Compute BERTscore for the two lists
        print(f"Length of preds={len(preds)} and len of refs = {len(refs)}\n")
        bertscore = load("bertscore")
        results = bertscore.compute(predictions=preds, references=refs, lang="en")

        # Compute the averages from the lists in the results dictionary.
        avg_precision = sum(results["precision"]) / len(results["precision"])
        avg_recall    = sum(results["recall"]) / len(results["recall"])
        avg_f1        = sum(results["f1"]) / len(results["f1"])

        # Return the average metrics.
        return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}
        

