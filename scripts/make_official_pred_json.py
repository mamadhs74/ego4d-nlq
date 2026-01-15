
import json

from pathlib import Path



gt_path = "/home/moho1597/ego4d_data/v2/annotations/nlq_val.json"

out_path = "/home/moho1597/ego4d_data/official_nlq_val_pred_middle20.json"



with open(gt_path, "r") as f:

    gt = json.load(f)



results = []

count = 0



for video in gt["videos"]:

    for clip in video["clips"]:

        clip_uid = clip["clip_uid"]



        # Clip-level time span (used only to define a baseline window)

        clip_start = clip.get("clip_start_sec", 0.0)

        clip_end = clip.get("clip_end_sec", None)

        if clip_end is None:

            # fall back: if missing, skip

            continue



        clip_len = max(0.0, clip_end - clip_start)

        pred_start = clip_start + 0.4 * clip_len

        pred_end   = clip_start + 0.6 * clip_len



        # Provide 3 identical windows because evaluator uses top-3 mean IoU

        pred_times = [[pred_start, pred_end], [pred_start, pred_end], [pred_start, pred_end]]



        for ann in clip["annotations"]:

            ann_uid = ann["annotation_uid"]

            lqs = ann.get("language_queries", [])

            for q_idx in range(len(lqs)):

                results.append({

                    "clip_uid": clip_uid,

                    "annotation_uid": ann_uid,

                    "query_idx": q_idx,

                    "predicted_times": pred_times

                })

                count += 1



pred = {

    "version": "1.0",

    "challenge": "ego4d_nlq_challenge",

    "results": results

}



Path(out_path).parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:

    json.dump(pred, f)



print("Wrote:", out_path)

print("Total predictions:", count)

print("Example item:", results[0])
