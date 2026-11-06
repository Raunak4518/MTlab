from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import mlflow

from mtlab.experiments.tracking import start_experiment_run, log_dataset_info
from mtlab.smt.phrase_based import train_phrase_table, train_language_model, save_language_model


def main():
    parser = argparse.ArgumentParser(description="Train a Python phrase-based SMT system from MGIZA alignments.")
    parser.add_argument("--manifest", default="D:/mtlab_work/python_pipeline/alignment_manifest.json")
    parser.add_argument("--model", choices=["ibm1", "ibm2", "hmm", "ibm3", "ibm4", "ibm5"], default="ibm4")
    parser.add_argument("--direction", choices=["en-hi", "hi-en"], default="en-hi")
    parser.add_argument("--output-dir", default="D:/mtlab_work/python_pipeline/phrase_smt")
    parser.add_argument("--max-phrase-len", type=int, default=7)
    parser.add_argument("--lm-order", type=int, default=3)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    runs = manifest["runs"]

    reverse_direction = "hi-en" if args.direction == "en-hi" else "en-hi"
    forward = next((r for r in runs if r["model"] == args.model and r["direction"] == args.direction), None)
    reverse = next((r for r in runs if r["model"] == args.model and r["direction"] == reverse_direction), None)
    if forward is None or reverse is None:
        raise RuntimeError(f"Manifest does not contain both {args.model} directions")

    train_en = Path(metadata["train_en"])
    train_hi = Path(metadata["train_hi"])
    if args.direction == "en-hi":
        train_source, train_target = train_en, train_hi
    else:
        train_source, train_target = train_hi, train_en
    out = Path(args.output_dir) / args.model / args.direction
    out.mkdir(parents=True, exist_ok=True)
    phrase_db = out / "phrase_table.sqlite"
    lm_path = out / "target_lm.pkl"

    with start_experiment_run(
        "02-phrase-based-smt", f"python-pbsmt-{args.model}-{args.direction}",
        params={
            "alignment_model": args.model,
            "direction": args.direction,
            "training_pairs": metadata["training_pairs"],
            "dataset_scope": metadata["scope"],
            "max_phrase_len": args.max_phrase_len,
            "language_model": "interpolated Kneser-Ney",
            "language_model_order": args.lm_order,
            "decoder": "coverage-stack phrase decoder",
            "symmetrization": "grow-diag-final-and",
        },
        tags={
            "research_area": "statistical-mt",
            "system": "phrase-based-smt",
            "alignment_model": args.model,
            "direction": args.direction,
            "dataset_scope": metadata["scope"],
        },
    ) as run:
        log_dataset_info("train", "Samanantar", "train", metadata["training_pairs"], train_en.parent)
        mlflow.set_tags({
            "forward_alignment_run_id": forward["run_id"],
            "reverse_alignment_run_id": reverse["run_id"],
            "phrase_table_path": str(phrase_db),
            "language_model_path": str(lm_path),
        })

        started = time.perf_counter()
        phrase_stats = train_phrase_table(
            train_source, train_target,
            forward["a3_path"], reverse["a3_path"],
            phrase_db, args.max_phrase_len,
        )
        phrase_seconds = time.perf_counter() - started

        started = time.perf_counter()
        lm = train_language_model(train_target, args.lm_order)
        save_language_model(lm, lm_path)
        lm_seconds = time.perf_counter() - started

        model_info = {
            "alignment_model": args.model,
            "direction": args.direction,
            "phrase_table": str(phrase_db),
            "language_model": str(lm_path),
            "training_pairs": phrase_stats["sentence_pairs"],
            "unique_phrase_pairs": phrase_stats["phrase_pairs"],
        }
        info_path = out / "model_info.json"
        info_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

        mlflow.log_metrics({
            "phrase_extraction_seconds": phrase_seconds,
            "language_model_seconds": lm_seconds,
            "unique_phrase_pairs": phrase_stats["phrase_pairs"],
        })
        mlflow.log_artifact(str(lm_path))
        mlflow.log_artifact(str(info_path))

    print(f"Phrase-based SMT trained: {out}")
    print(f"Unique phrase pairs: {phrase_stats['phrase_pairs']:,}")
    print(f"Phrase table: {phrase_db}")
    print(f"Language model: {lm_path}")


if __name__ == "__main__":
    main()
