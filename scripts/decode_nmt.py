from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from mtlab.nmt.models import build_nmt_model, greedy_decode


def main():
    parser = argparse.ArgumentParser(description="Generate translations from a trained NMT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="Tokenized source sentences, one per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = build_nmt_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    processor = spm.SentencePieceProcessor(model_file=checkpoint["spm_model"])
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_ids = []
    batch_lines = []
    total = 0

    def flush(out_file):
        nonlocal batch_ids, batch_lines, total
        if not batch_ids:
            return
        max_len = max(map(len, batch_ids))
        source_batch = torch.full(
            (len(batch_ids), max_len), config["pad_id"], dtype=torch.long
        )
        for i, ids in enumerate(batch_ids):
            source_batch[i, :len(ids)] = torch.tensor(ids)
        source_batch = source_batch.to(device)
        generated = greedy_decode(
            model, source_batch,
            config["bos_id"], config["eos_id"],
            max_length=args.max_length,
            pad_id=config["pad_id"],
        ).cpu().tolist()
        for ids in generated:
            cleaned = []
            for token_id in ids[1:]:
                if token_id == config["eos_id"]:
                    break
                if token_id != config["pad_id"] and token_id != config["bos_id"]:
                    cleaned.append(token_id)
            out_file.write(processor.decode(cleaned) + "\n")
            total += 1
        batch_ids.clear()
        batch_lines.clear()

    with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as output:
        for line in source:
            ids = [config["bos_id"], *processor.encode(line.strip(), out_type=int), config["eos_id"]]
            batch_ids.append(ids[:args.max_length])
            batch_lines.append(line)
            if len(batch_ids) >= args.batch_size:
                flush(output)
        flush(output)

    print(f"Generated {total} translations: {output_path}")


if __name__ == "__main__":
    main()
