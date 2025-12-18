import json

from pathlib import Path



src_path = "/home/moho1597/ego4d_data/v2/annotations/nlq_train.json"

out_path = "/home/moho1597/projects/ego4d-nlq/data/nlq_train_samples.json"



with open(src_path, "r") as f:

    data = json.load(f)



samples = []

skipped = 0



for video in data["videos"]:

    video_uid = video["video_uid"]

    for clip in video["clips"]:

        clip_uid = clip["clip_uid"]

        for ann in clip["annotations"]:

            for lq in ann.get("language_queries", []):

                if "query" not in lq:

                    skipped += 1

                    continue

                samples.append({

                    "video_uid": video_uid,

                    "clip_uid": clip_uid,

                    "query": lq["query"],

                    "clip_start_sec": lq["clip_start_sec"],

                    "clip_end_sec": lq["clip_end_sec"],

                })



Path(out_path).parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:

    json.dump(samples, f, indent=2)



print("Extracted:", len(samples), "samples | Skipped:", skipped)


