import pandas as pd

files = ["results/experiments/v1_yolo11n_smoke_test/results.csv",
    "results/experiments/v1_yolo11n_30ep_baseline/results.csv",
    "results/experiments/arch_baseline_scratch_30ep/results.csv"]
for i in files:
    df = pd.read_csv(i)

    print(df[[
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)"
    ]].to_string(index=False))