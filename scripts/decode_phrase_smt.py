from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtlab.smt.phrase_based import load_language_model, PhraseBasedDecoder
from mtlab.experiments.tracking import start_experiment_run
import mlflow
import time


def main():
    parser = argparse.ArgumentParser(description="Decode with the Python phrase-based SMT model.")
    parser.add_argument("--model-info", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--max-phrase-len", type=int, default=7)
    parser.add_argument("--lm-weight", type=float, default=0.25)
    parser.add_argument("--distortion-weight", type=float, default=0.3)
    parser.add_argument("--max-distortion", type=int, default=6)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--target-lang", default="hi")
    parser.add_argument("--no-detokenize", action="store_true")
    args = parser.parse_args()

    info = json.loads(Path(args.model_info).read_text(encoding="utf-8"))
    lm = load_language_model(info["language_model"])
    decoder = PhraseBasedDecoder(
        info["phrase_table"], lm,
        beam_size=args.beam_size,
        max_phrase_len=args.max_phrase_len,
        lm_weight=args.lm_weight,
        distortion_weight=args.distortion_weight,
        max_distortion=args.max_distortion,
    )

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detokenizer = None
    if not args.no_detokenize:
        try:
            from sacremoses import MosesDetokenizer
            detokenizer = MosesDetokenizer(lang=args.target_lang)
        except ImportError:
            print("sacremoses not installed; writing tokenized hypotheses.")

    started = time.perf_counter()
    try:
        with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as output:
            for index, line in enumerate(source, start=1):
                hypothesis = decoder.decode(line.strip().split())
                if detokenizer is not None:
                    hypothesis = detokenizer.detokenize(hypothesis.split())
                output.write(hypothesis + "\n")
                if index % 100 == 0:
                    print(f"Decoded {index} sentences", flush=True)
    finally:
        decoder.close()
    elapsed = time.perf_counter() - started
    sentence_count = sum(1 for _ in output_path.open(encoding="utf-8"))
    with start_experiment_run(
        "02-phrase-based-smt", args.run_name or f"decode-{info['alignment_model']}-en-hi",
        params={
            "alignment_model": info["alignment_model"],
            "direction": info.get("direction", "en-hi"),
            "decoder": "coverage-stack phrase decoder",
            "beam_size": args.beam_size,
            "lm_weight": args.lm_weight,
            "distortion_weight": args.distortion_weight,
            "max_distortion": args.max_distortion,
            "sentences": sentence_count,
            "detokenized": not args.no_detokenize and detokenizer is not None,
        },
        tags={"research_area": "statistical-mt", "stage": "decoding", "system": "python-pbsmt"},
    ):
        mlflow.log_metrics({
            "decoding_seconds": elapsed,
            "sentences_per_second": sentence_count / max(elapsed, 1e-9),
        })
        mlflow.log_artifact(str(output_path))
    print(f"Hypotheses written to {output_path}")


if __name__ == "__main__":
    main()
