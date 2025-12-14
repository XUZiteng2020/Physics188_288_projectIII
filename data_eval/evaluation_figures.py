import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

#Produces and saves heatmap locally
def save_heatmap(summary_csv, outpath, value_col="verif_r2", title=None):
    df = pd.read_csv(summary_csv)

    pivot = df.pivot_table(index="dataset", columns="config_id", values=value_col, aggfunc="max")

    plt.figure(figsize=(10, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=value_col)
    plt.title(title if title else f"{value_col} heatmap (dataset x config)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_best_config_bar(summary_csv, outpath, value_col="verif_r2", title=None):
    df = pd.read_csv(summary_csv)
    best = df.sort_values(value_col, ascending=False).groupby("dataset", as_index=False).first()

    plt.figure(figsize=(8, 4))
    plt.bar(best["dataset"], best[value_col])
    plt.ylabel(value_col)
    plt.title(title if title else f"Best {value_col} per dataset")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_feature_set_drop_plot(all_csv, outpath):
    df = pd.read_csv(all_csv)

    # rows=config_id, cols=dataset (full/longlist/shortlist), values=verif_r2
    piv = df.pivot_table(index="config_id", columns="dataset", values="verif_r2", aggfunc="mean")

    for col in ["full", "longlist", "shortlist"]:
        if col not in piv.columns:
            raise ValueError(f"Missing dataset column '{col}' in {all_csv}. Found: {list(piv.columns)}")

    drop_full_to_short = (piv["full"] - piv["shortlist"]).sort_values(ascending=False)
    drop_full_to_long  = (piv["full"] - piv["longlist"]).loc[drop_full_to_short.index]

    plt.figure(figsize=(10, 4))
    x = np.arange(len(drop_full_to_short))
    plt.bar(x - 0.2, drop_full_to_short.values, width=0.4, label="full - shortlist")
    plt.bar(x + 0.2, drop_full_to_long.values,  width=0.4, label="full - longlist")
    plt.xticks(x, drop_full_to_short.index, rotation=35, ha="right")
    plt.ylabel("Δ verification R2")
    plt.title("Feature-set sensitivity by configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_optimizer_comparison_plot(data_eval_dir, outpath):
    files = [
        "nn_config_Adam_Baseline_summary.csv",
        "nn_config_MiniBatch_SGD_Momentum_summary.csv",
        "nn_config_Pure_SGD_summary.csv",
        "nn_config_Standard_Batch_GD_summary.csv",
    ]

    rows = []
    for f in files:
        path = os.path.join(data_eval_dir, f)
        d = pd.read_csv(path).iloc[0]
        rows.append({
            "label": str(d["config_id"]),
            "val_r2": float(d["val_r2"]),
            "verif_r2": float(d["verif_r2"]),
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(df))
    plt.bar(x - 0.2, df["val_r2"], width=0.4, label="validation R2")
    plt.bar(x + 0.2, df["verif_r2"], width=0.4, label="verification R2")
    plt.xticks(x, df["label"], rotation=25, ha="right")
    plt.ylabel("R2")
    plt.title("Optimizer comparison (summary runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    root = os.path.dirname(os.path.dirname(__file__))
    data_eval_dir = "data_eval"
    figures_dir = os.path.join(root, "figures")
    ensure_dir(figures_dir)

    summary_csv = os.path.join(root, "data_eval", "nn_all_datasets_all_configs_summary.csv")

    save_heatmap( #Ver R2
        summary_csv,
        os.path.join(figures_dir, "eval_heatmap_verif_r2.png"),
        value_col="verif_r2",
        title="Verification R2 across datasets/configs"  
    )

    save_best_config_bar(
        summary_csv,
        os.path.join(figures_dir, "eval_best_verif_r2_per_dataset.png"),
        value_col="verif_r2",
        title="Best verification R2 per dataset"
    )

    save_heatmap( #Val R2
        summary_csv,
        os.path.join(figures_dir, "eval_heatmap_val_r2.png"),
        value_col="val_r2",
        title="Validation R2 across datasets/configs"
    )

    save_feature_set_drop_plot(os.path.join(data_eval_dir, "nn_all_datasets_all_configs_summary.csv"),os.path.join(figures_dir, "eval_feature_set_drop.png"),)

    save_optimizer_comparison_plot(data_eval_dir, os.path.join(figures_dir, "eval_optimizer_comparison.png"),)


    print("Saved evaluation summary figures into figures/")

if __name__ == "__main__":
    main()
