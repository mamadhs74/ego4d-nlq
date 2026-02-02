#!/usr/bin/env python3
import argparse, json, os

def load_preds(path):
    with open(path, "r") as f:
        d = json.load(f)

    # common shapes:
    # 1) {"results":[...]}
    # 2) [...] (list)
    # 3) {"predictions":[...]} / {"preds":[...]}
    if isinstance(d, dict):
        for k in ["results", "predictions", "preds"]:
            if k in d and isinstance(d[k], list):
                return d[k]
    if isinstance(d, list):
        return d
    raise ValueError(f"Unrecognized prediction JSON structure in {path}: type={type(d)} keys={list(d.keys()) if isinstance(d, dict) else None}")

def to_official(items):
    out = []
    for it in items:
        # Be tolerant to key names
        clip_uid = it.get("clip_uid") or it.get("clip_id")
        ann_uid  = it.get("annotation_uid") or it.get("annotation_id")
        qidx     = it.get("query_idx") if "query_idx" in it else it.get("query_index", 0)

        # VSLNet-style sometimes uses "predicted_times" already
        times = it.get("predicted_times") or it.get("pred_times") or it.get("predictions")

        if clip_uid is None or ann_uid is None or times is None:
            continue

        out.append({
            "clip_uid": clip_uid,
            "annotation_uid": ann_uid,
            "query_idx": int(qidx),
            "predicted_times": times,
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="Input model predictions (e.g., vslnet_XXXX_test_result.json)")
    ap.add_argument("--out_path", required=True, help="Output official-format JSON")
    ap.add_argument("--version", default="1.0")
    args = ap.parse_args()

    items = load_preds(args.pred_path)
    results = to_official(items)

    payload = {"version": args.version, "results": results}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(payload, f)

    print(f"Wrote: {args.out_path}")
    print(f"Total predictions: {len(results)}")
    if results:
        print("Example item:", results[0])

if __name__ == "__main__":
    main()

