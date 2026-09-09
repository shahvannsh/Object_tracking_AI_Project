"""
MOTA / IDF1 evaluation against ground truth in MOT format.
Usage: python evaluation/metrics.py --gt data/annotations/gt.csv --pred outputs/logs/tracks.csv
"""
import argparse
import pandas as pd

try:
    import motmetrics as mm
except ImportError:
    mm = None


def compute_metrics(gt_path: str, pred_path: str):
    if mm is None:
        raise ImportError("Run: pip install motmetrics")

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt["frame"]).union(pred["frame"]))
    for frame in frames:
        gt_frame = gt[gt["frame"] == frame]
        pred_frame = pred[pred["frame"] == frame]

        gt_ids = gt_frame["track_id"].tolist()
        pred_ids = pred_frame["track_id"].tolist()

        gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].values
        pred_boxes = pred_frame[["x1", "y1", "x2", "y2"]].values

        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "idf1"], name="tracking")
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    compute_metrics(args.gt, args.pred)
