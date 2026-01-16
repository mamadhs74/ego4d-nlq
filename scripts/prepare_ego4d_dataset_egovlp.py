

#! /usr/bin/env python

"""

Prepare Ego4D NLQ splits for VSLNet when features are already clip-level (EgoVLP).

- Writes train/val/test JSONs in VSLNet format (same as original script)

- Copies (or symlinks) clip_uid.pt features directly into clip_feature_save_path

"""



import argparse

import json

import math

import os

from pathlib import Path



import torch

import tqdm



from prepare_ego4d_dataset import reformat_data  # reuse existing formatter





def main(args):

    all_clip_uids = set()



    # 1) write reformatted jsons

    for split in ("train", "val"):

        read_path = args[f"input_{split}_split"]

        print(f"Reading [{split}]: {read_path}")

        with open(read_path, "r") as f:

            raw = json.load(f)



        data_split, _clip_video_map = reformat_data(raw, split == "test")



        num_instances = sum(len(v["sentences"]) for v in data_split.values())

        print(f"# {split}: {num_instances}")



        os.makedirs(args["output_save_path"], exist_ok=True)

        out_json = os.path.join(args["output_save_path"], f"{split}.json")

        print(f"Writing [{split}]: {out_json}")

        with open(out_json, "w") as f:

            json.dump(data_split, f)



        # collect clip uids

        for clip_uid in data_split.keys():

            all_clip_uids.add(clip_uid)



    # 2) copy/symlink clip-level features

    os.makedirs(args["clip_feature_save_path"], exist_ok=True)

    feature_sizes = {}



    feat_root = Path(args["video_feature_read_path"])

    out_root = Path(args["clip_feature_save_path"])



    print(f"Copying clip-level features from: {feat_root}")

    print(f"Saving clip features to: {out_root}")



    for clip_uid in tqdm.tqdm(sorted(all_clip_uids), desc="Linking features"):

        src = feat_root / f"{clip_uid}.pt"

        dst = out_root / f"{clip_uid}.pt"



        if not src.exists():

            raise FileNotFoundError(f"Missing feature for clip_uid: {clip_uid} at {src}")



        if dst.exists():

            continue



        if args["symlink"]:

            dst.symlink_to(src)

        else:

            # copy tensor (safe but slower)

            x = torch.load(src, map_location="cpu")

            torch.save(x, dst)



        # read shape once (cheap if symlink, still loads)

        x = torch.load(src, map_location="cpu")

        feature_sizes[clip_uid] = x.shape[0]



    with open(out_root / "feature_shapes.json", "w") as f:

        json.dump(feature_sizes, f)



    print("Done. feature_shapes.json written.")





if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--input_train_split", required=True)

    p.add_argument("--input_val_split", required=True)

    p.add_argument("--output_save_path", required=True)

    # For EgoVLP this is actually the clip-level feature dir

    p.add_argument("--video_feature_read_path", required=True)

    p.add_argument("--clip_feature_save_path", required=True)

    p.add_argument("--symlink", action="store_true", help="Symlink instead of copying")



    a = vars(p.parse_args())

    main(a)


