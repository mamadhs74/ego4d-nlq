
import json



val_path = "data/nlq_val_samples.json"

out_path = "results/nlq_val_pred_middle20.json"



with open(val_path, "r") as f:

    samples = json.load(f)



predictions = []



for s in samples:

    clip_start = s["clip_start_sec"]

    clip_end = s["clip_end_sec"]

    clip_len = clip_end - clip_start



    pred_start = clip_start + 0.4 * clip_len

    pred_end = clip_start + 0.6 * clip_len



    predictions.append({

        "video_uid": s["video_uid"],

        "clip_uid": s["clip_uid"],

        "query": s["query"],

        "gt_start": clip_start,

        "gt_end": clip_end,

        "pred_start": pred_start,

        "pred_end": pred_end

    })



with open(out_path, "w") as f:

    json.dump(predictions, f, indent=2)



print("Saved predictions to", out_path)


