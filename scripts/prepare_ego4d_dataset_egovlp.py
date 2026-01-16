#! /usr/bin/env python

"""

Prepare Ego4D NLQ splits for VSLNet when features are already clip-level (EgoVLP).

Works even when EgoVLP features only partially overlap with Ego4D v2 annotations.



Outputs:

- <output_save_path>/train.json and val.json (VSLNet format, filtered to clips with features)

- <clip_feature_save_path>/*.pt (symlinks or copies of clip-level features)

- <clip_feature_save_path>/feature_shapes.json (clip_uid -> #frames)



Note:

- This script intentionally prepares only train+val (no test) for the safe baseline.

"""



import argparse

import json

import os

from pathlib import Path



import torch

import tqdm



# reuse the official formatter from the original script

from prepare_ego4d_dataset import reformat_data





def main(args):

    feat_root = Path(args["video_feature_read_path"]).expanduser().resolve()

    out_json_root = Path(args["output_save_path"]).expanduser().resolve()

    out_feat_root = Path(args["clip_feature_save_path"]).expanduser().resolve()



    if not feat_root.exists():

        raise FileNotFoundError(f"Feature directory not found: {feat_root}")



    # Available feature files: clip_uid.pt

    available = {p.stem for p in feat_root.glob("*.pt")}

    print("Feature root:", feat_root)

    print("Available feature files:", len(available))



    os.makedirs(out_json_root, exist_ok=True)

    os.makedirs(out_feat_root, exist_ok=True)



    all_clip_uids = set()

    split_paths = {

        "train": args["input_train_split"],

        "val": args["input_val_split"],

    }



    # 1) Write filtered train/val json

    for split, read_path in split_paths.items():

        print(f"Reading [{split}]: {read_path}")

        with open(read_path, "r") as f:

            raw = json.load(f)



        data_split, _clip_video_map = reformat_data(raw, False)  # test=False for train/val



        # Filter to only clips with available features

        filtered = {}

        kept = 0

        dropped = 0

        for clip_uid, item in data_split.items():

            if clip_uid in available:

                filtered[clip_uid] = item

                all_clip_uids.add(clip_uid)

                kept += 1

            else:

                dropped += 1



        num_instances = sum(len(v["sentences"]) for v in filtered.values())

        print(f"# {split} instances (after filter): {num_instances}")

        print(f"[{split}] kept clips: {kept} | dropped (no features): {dropped}")



        out_json = out_json_root / f"{split}.json"

        print(f"Writing [{split}]: {out_json}")

        with open(out_json, "w") as f:

            json.dump(filtered, f)



    # 2) Link/copy features for kept clip_uids

    feature_sizes = {}

    print("Preparing clip-level features into:", out_feat_root)

    for clip_uid in tqdm.tqdm(sorted(all_clip_uids), desc="Linking features"):

        src = feat_root / f"{clip_uid}.pt"

        if not src.exists():

            # Should not happen because we filtered, but keep it robust

            continue



        dst = out_feat_root / f"{clip_uid}.pt"

        if dst.exists():

            continue



        if args["symlink"]:

            dst.symlink_to(src)

        else:

            x = torch.load(src, map_location="cpu")

            torch.save(x, dst)



        # record shape

        x = torch.load(src, map_location="cpu")

        feature_sizes[clip_uid] = int(x.shape[0])



    shapes_path = out_feat_root / "feature_shapes.json"

    with open(shapes_path, "w") as f:

        json.dump(feature_sizes, f)



    print("Done.")

    print("Wrote feature shapes:", shapes_path)

    print("Total clips linked:", len(feature_sizes))





if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--input_train_split", required=True)

    p.add_argument("--input_val_split", required=True)

    p.add_argument("--output_save_path", required=True)

    # For EgoVLP this is the clip-level feature dir (clip_uid.pt)

    p.add_argument("--video_feature_read_path", required=True)

    p.add_argument("--clip_feature_save_path", required=True)

    p.add_argument("--symlink", action="store_true")



    args = vars(p.parse_args())

    main(args)


