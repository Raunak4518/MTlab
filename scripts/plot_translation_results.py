from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Create presentation-ready translation metric comparisons.")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--output-dir", default="reports/figures/translation")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(tables_dir.glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append({
            "label": f"{data['system']}\n{data['direction']}\n{data['dataset_split']}",
            "system": data["system"],
            "direction": data["direction"],
            "dataset_split": data["dataset_split"],
            **data["scores"],
        })

    if not records:
        raise FileNotFoundError(f"No *_metrics.json files found in {tables_dir}")

    csv_path = output_dir / "translation_comparison.csv"
    fields = sorted({key for row in records for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    metrics = ["bleu", "spbleu", "chrf++", "ter", "comet"]
    for metric in metrics:
        available = [row for row in records if metric in row]
        if not available:
            continue
        labels = [row["label"] for row in available]
        values = [row[metric] for row in available]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
        bars = ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Translation system comparison: {metric.upper()}")
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric.replace('+', 'p')}_comparison.png", dpi=300)
        plt.close(fig)

    print(f"Comparison table: {csv_path}")
    print(f"Figures: {output_dir}")


if __name__ == "__main__":
    main()
