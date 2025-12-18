
import json



pred_path = "results/nlq_val_pred_middle20.json"



with open(pred_path, "r") as f:

    preds = json.load(f)



def tiou(a_start, a_end, b_start, b_end):

    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))

    union = max(a_end, b_end) - min(a_start, b_start)

    return inter / union if union > 0 else 0.0



scores = [

    tiou(p["gt_start"], p["gt_end"], p["pred_start"], p["pred_end"])

    for p in preds

]



def recall_at(th):

    return sum(s >= th for s in scores) / len(scores)



print("Mean tIoU:", sum(scores) / len(scores))

print("Recall@0.3:", recall_at(0.3))

print("Recall@0.5:", recall_at(0.5))


