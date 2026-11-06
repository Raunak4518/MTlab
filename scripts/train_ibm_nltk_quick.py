from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from contextlib import suppress
from itertools import zip_longest
from pathlib import Path

import nltk
import cloudpickle
from nltk.translate import AlignedSent
from nltk.translate.ibm4 import IBMModel4
from nltk.translate.ibm5 import IBMModel5


def frequency_rank_classes(sentences: list[list[str]], n_classes: int = 50) -> dict[str, int]:
    """Deterministic frequency-rank classes (not Brown/mkcls clusters)."""
    counts = Counter(token for sent in sentences for token in sent)
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    if not ordered:
        return {}
    return {
        token: min(n_classes - 1, rank * n_classes // len(ordered))
        for rank, token in enumerate(ordered)
    }


def load_pairs(src_path: Path, tgt_path: Path, max_pairs: int, max_len: int):
    pairs: list[tuple[list[str], list[str]]] = []
    skipped_empty = 0
    skipped_long = 0
    with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
        for line_no, row in enumerate(zip_longest(src_f, tgt_f), start=1):
            src_line, tgt_line = row
            if src_line is None or tgt_line is None:
                raise ValueError(f"Parallel files have different line counts at line {line_no}")
            source = src_line.strip().split()
            target = tgt_line.strip().split()
            if not source or not target:
                skipped_empty += 1
                continue
            if max_len and (len(source) > max_len or len(target) > max_len):
                skipped_long += 1
                continue
            pairs.append((source, target))
            if max_pairs and len(pairs) >= max_pairs:
                break
    if not pairs:
        raise ValueError("No usable sentence pairs. Check the tokenized train.en/train.hi files.")
    return pairs, skipped_empty, skipped_long


def make_corpus(pairs: list[tuple[list[str], list[str]]], direction: str):
    if direction == "en-hi":
        # NLTK uses mots=source and words=target.
        return [AlignedSent(words=hi, mots=en) for en, hi in pairs]
    return [AlignedSent(words=en, mots=hi) for en, hi in pairs]


def links_for(pair: AlignedSent) -> list[tuple[int, int]]:
    """Convert NLTK (target, source) links to Pharaoh (source, target).

    NLTK may represent unaligned words with None on either side. Such
    entries are not links and must be excluded from Pharaoh/A3 output.
    """
    links = []
    for target_pos, source_pos in pair.alignment:
        if target_pos is None or source_pos is None:
            continue
        links.append((int(source_pos), int(target_pos)))
    return sorted(links)


def write_outputs(corpus, direction: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pharaoh_path = out_dir / "alignments.pharaoh"
    a3_path = out_dir / "alignments.a3"

    with pharaoh_path.open("w", encoding="utf-8", newline="\n") as pharaoh, \
         a3_path.open("w", encoding="utf-8", newline="\n") as a3:
        for sent_no, pair in enumerate(corpus, start=1):
            source_tokens = list(pair.mots)
            target_tokens = list(pair.words)
            links = links_for(pair)
            pharaoh.write(" ".join(f"{s}-{t}" for s, t in links) + "\n")

            target_to_source: dict[int, list[int]] = {j: [] for j in range(len(target_tokens))}
            for source_pos, target_pos in links:
                if 0 <= source_pos < len(source_tokens) and 0 <= target_pos < len(target_tokens):
                    target_to_source[target_pos].append(source_pos + 1)  # A3 is 1-based.
            groups = [f"{word} ({{ {' '.join(map(str, sorted(target_to_source[j])))} }})"
                      for j, word in enumerate(target_tokens)]
            a3.write(
                f"# Sentence pair ({sent_no}) source length {len(source_tokens)} "
                f"target length {len(target_tokens)} alignment score : 0\n"
            )
            a3.write(" ".join(source_tokens) + "\n")
            a3.write(" ".join(groups) + "\n")
    return pharaoh_path, a3_path


def mlflow_start(run_name: str, params: dict, tags: dict):
    """Best-effort MLflow: training still succeeds if the server is unavailable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        mlflow.set_experiment("01-word-alignment")
        active = mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        return mlflow, active.info.run_id
    except Exception as exc:
        print(f"[MLflow warning] Tracking unavailable; continuing locally: {exc}", flush=True)
        with suppress(Exception):
            import mlflow
            if mlflow.active_run():
                mlflow.end_run()
        return None, None


def mlflow_finish(mlflow, metrics: dict, artifact_paths: list[Path]):
    if mlflow is None:
        return
    try:
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            mlflow.log_artifact(str(path))
    except Exception as exc:
        print(f"[MLflow warning] Could not log every artifact: {exc}", flush=True)
    finally:
        with suppress(Exception):
            mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="Small-subset NLTK IBM4/IBM5 word alignment.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-en", default="data/processed/samanantar/en-hi/word/train.en")
    parser.add_argument("--train-hi", default="data/processed/samanantar/en-hi/word/train.hi")
    parser.add_argument("--models", default="ibm4,ibm5", help="Comma-separated: ibm4,ibm5")
    parser.add_argument("--directions", default="en-hi,hi-en", help="Comma-separated: en-hi,hi-en")
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="Use 5-10 for a quick NLTK reference run; 0 means all (not recommended).")
    parser.add_argument("--max-sentence-length", type=int, default=10,
                        help="Skip pairs longer than this many whitespace tokens on either side; 0 disables.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--classes", type=int, default=50)
    parser.add_argument("--output-dir", default="artifacts/ibm_nltk")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if args.max_pairs < 0 or args.iterations < 1 or args.classes < 1:
        parser.error("--max-pairs must be >= 0; --iterations and --classes must be >= 1")
    models = [v.strip().lower() for v in args.models.split(",") if v.strip()]
    directions = [v.strip().lower() for v in args.directions.split(",") if v.strip()]
    if not models or any(m not in {"ibm4", "ibm5"} for m in models):
        parser.error("--models accepts ibm4 and/or ibm5")
    if not directions or any(d not in {"en-hi", "hi-en"} for d in directions):
        parser.error("--directions accepts en-hi and/or hi-en")

    root = Path(args.project_root).resolve()
    train_en = (root / args.train_en).resolve()
    train_hi = (root / args.train_hi).resolve()
    for path in (train_en, train_hi):
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {path}")
    pairs, skipped_empty, skipped_long = load_pairs(
        train_en, train_hi, args.max_pairs, args.max_sentence_length
    )
    out_root = (root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    subset_en = out_root / "training_subset.en"
    subset_hi = out_root / "training_subset.hi"
    subset_en.write_text("\n".join(" ".join(en) for en, _ in pairs) + "\n", encoding="utf-8")
    subset_hi.write_text("\n".join(" ".join(hi) for _, hi in pairs) + "\n", encoding="utf-8")

    print(f"NLTK {nltk.__version__}; usable pairs={len(pairs)}; "
          f"skipped empty={skipped_empty}; skipped over-length={skipped_long}", flush=True)
    print("Important: IBM5 internally trains IBM4 first. This is a tiny reference run, not a full-corpus result.", flush=True)

    run_manifest = []
    for model_name in models:
        for direction in directions:
            corpus = make_corpus(pairs, direction)
            src_classes = frequency_rank_classes([list(s.mots) for s in corpus], args.classes)
            tgt_classes = frequency_rank_classes([list(s.words) for s in corpus], args.classes)
            out_dir = out_root / model_name / direction
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / "model.pkl"
            params = {
                "engine": "NLTK", "model": model_name.upper(), "direction": direction,
                "training_pairs": len(pairs), "iterations": args.iterations,
                "classes": args.classes, "max_sentence_length": args.max_sentence_length,
                "class_assignment": "frequency-rank buckets; not mkcls",
                "dataset": "Samanantar tokenized train subset",
            }
            tags = {"research_area": "word-alignment", "engine": "nltk",
                    "model": model_name, "direction": direction,
                    "dataset_scope": "tiny-reference-subset"}
            tracker, run_id = (None, None) if args.no_mlflow else mlflow_start(
                f"nltk-{model_name}-{direction}", params, tags
            )

            started = time.perf_counter()
            if model_name == "ibm4":
                model = IBMModel4(corpus, args.iterations, src_classes, tgt_classes)
            else:
                model = IBMModel5(corpus, args.iterations, src_classes, tgt_classes)
            elapsed = time.perf_counter() - started

            # NLTK updates corpus[i].alignment to its best Viterbi alignment.
            with model_path.open("wb") as f:
                cloudpickle.dump({"model": model, "source_classes": src_classes,
                                  "target_classes": tgt_classes, "model_name": model_name,
                                  "direction": direction}, f, protocol=pickle.HIGHEST_PROTOCOL)
            pharaoh_path, a3_path = write_outputs(corpus, direction, out_dir)
            metrics = {
                "training_seconds": elapsed,
                "training_pairs": len(corpus),
                "training_pairs_per_second": len(corpus) / max(elapsed, 1e-9),
                "alignment_links": sum(len(links_for(s)) for s in corpus),
            }
            if tracker is not None:
                mlflow_finish(tracker, metrics, [model_path, pharaoh_path, a3_path])
            print(f"DONE {model_name.upper()} {direction}: {elapsed:.2f}s -> {out_dir}", flush=True)
            run_manifest.append({
                "model": model_name, "direction": direction,
                "run_id": run_id, "model_path": str(model_path),
                "pharaoh_path": str(pharaoh_path), "a3_path": str(a3_path),
                "training_pairs": len(pairs),
            })

    manifest_path = out_root / "alignment_manifest.json"
    manifest_path.write_text(json.dumps({
        "metadata": {
            "train_en": str(subset_en), "train_hi": str(subset_hi),
            "training_pairs": len(pairs), "scope": "tiny-nltk-reference-subset",
            "engine": "NLTK", "nltk_version": nltk.__version__,
            "skipped_empty": skipped_empty, "skipped_over_length": skipped_long,
        },
        "runs": run_manifest,
    }, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)
    print("Complete. NLTK alignments are available as Pharaoh text and Moses-style A3 files.", flush=True)


if __name__ == "__main__":
    main()
