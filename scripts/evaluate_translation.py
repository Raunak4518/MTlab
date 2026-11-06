from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import sacrebleu

from mtlab.experiments.tracking import start_experiment_run, log_dataset_info


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def main():
    parser = argparse.ArgumentParser(description="Evaluate translations with SacreBLEU metrics and optional COMET.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--spm-model", default="", help="Official FLORES SentencePiece model for spBLEU")
    parser.add_argument("--comet", action="store_true", help="Optionally run Unbabel COMET; downloads model on first use")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-gpus", type=int, default=0, help="Default 0 avoids using the 4 GB GPU for COMET.")
    parser.add_argument("--experiment", default="08-final-evaluation")
    parser.add_argument("--run-name", default="translation-evaluation")
    parser.add_argument("--direction", default="en-hi")
    parser.add_argument("--system", default="unspecified")
    parser.add_argument("--dataset-split", default="devtest")
    args = parser.parse_args()

    source = read_lines(args.source)
    reference = read_lines(args.reference)
    hypothesis = read_lines(args.hypothesis)
    if not (len(source) == len(reference) == len(hypothesis)):
        raise ValueError(
            f"Line counts differ: source={len(source)}, reference={len(reference)}, hypothesis={len(hypothesis)}"
        )
    if not source:
        raise ValueError("No evaluation sentences found")

    bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    chrf = sacrebleu.corpus_chrf(hypothesis, [reference], word_order=2)
    ter = sacrebleu.corpus_ter(hypothesis, [reference])

    scores = {
        "bleu": float(bleu.score),
        "chrf++": float(chrf.score),
        "ter": float(ter.score),
    }
    signatures = {
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
        "ter_signature": str(ter.get_signature()),
    }

    if args.spm_model:
        import sentencepiece as spm
        processor = spm.SentencePieceProcessor(model_file=args.spm_model)
        spm_hyp = [" ".join(processor.encode(x, out_type=str)) for x in hypothesis]
        spm_ref = [" ".join(processor.encode(x, out_type=str)) for x in reference]
        spbleu = sacrebleu.corpus_bleu(
            spm_hyp, [spm_ref], tokenize="none"
        )
        scores["spbleu"] = float(spbleu.score)
        signatures["spbleu_signature"] = str(spbleu.get_signature())

    if args.comet:
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                "COMET is not installed. Install it with: uv add --group experiment unbabel-comet"
            ) from exc
        model_path = download_model(args.comet_model)
        comet_model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(source, hypothesis, reference)
        ]
        prediction = comet_model.predict(comet_data, batch_size=8, gpus=args.comet_gpus)
        scores["comet"] = float(prediction.system_score)

    result = {
        "system": args.system,
        "direction": args.direction,
        "dataset_split": args.dataset_split,
        "sentences": len(source),
        "scores": scores,
        "signatures": signatures,
        "source": str(Path(args.source).resolve()),
        "reference": str(Path(args.reference).resolve()),
        "hypothesis": str(Path(args.hypothesis).resolve()),
    }

    output_dir = Path("reports/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.system}_{args.direction}_{args.dataset_split}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with start_experiment_run(
        args.experiment, args.run_name,
        params={
            "system": args.system,
            "direction": args.direction,
            "dataset_split": args.dataset_split,
            "evaluation_sentences": len(source),
            "evaluation_metrics": ",".join(scores.keys()),
        },
        tags={
            "research_area": "translation-evaluation",
            "system": args.system,
            "direction": args.direction,
            "dataset_split": args.dataset_split,
        },
    ):
        mlflow.log_metrics(scores)
        mlflow.log_artifact(str(result_path))
        mlflow.log_artifact(str(args.hypothesis))
        log_dataset_info(
            "test", "FLORES-200", args.dataset_split,
            len(source), Path(args.source).parent,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
