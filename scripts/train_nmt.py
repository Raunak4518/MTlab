from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from itertools import zip_longest

import mlflow
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from mtlab.experiments.tracking import start_experiment_run, log_dataset_info
from mtlab.nmt.models import build_nmt_model


PAD_ID, BOS_ID, EOS_ID = 3, 1, 2


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


class ParallelStreamDataset(IterableDataset):
    """Stream encoded sentence pairs to keep full-corpus NMT memory bounded."""

    def __init__(self, src_path, tgt_path, processor, max_pairs=0,
                 max_length=128, shuffle_buffer=0, seed=42):
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)
        self.processor = processor
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0
        src_count = _line_count(self.src_path)
        tgt_count = _line_count(self.tgt_path)
        if src_count != tgt_count:
            raise ValueError(
                f"Parallel files differ in length: {src_count} vs {tgt_count}"
            )
        self.pair_count = min(src_count, max_pairs) if max_pairs else src_count
        if self.pair_count == 0:
            raise ValueError("No parallel sentence pairs found")

    def __len__(self):
        return self.pair_count

    def _encode_pair(self, src, tgt, line_no):
        if not src.strip() or not tgt.strip():
            raise ValueError(f"Empty parallel sentence at line {line_no}")
        src_ids = [BOS_ID, *self.processor.encode(src.strip(), out_type=int), EOS_ID]
        tgt_ids = [BOS_ID, *self.processor.encode(tgt.strip(), out_type=int), EOS_ID]
        src_ids = src_ids[:self.max_length]
        tgt_ids = tgt_ids[:self.max_length]
        src_ids[-1] = EOS_ID
        tgt_ids[-1] = EOS_ID
        return src_ids, tgt_ids

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        buffer = []
        with self.src_path.open(encoding="utf-8") as src_file, self.tgt_path.open(encoding="utf-8") as tgt_file:
            for line_no, pair in enumerate(zip_longest(src_file, tgt_file), start=1):
                if line_no > self.pair_count:
                    break
                src, tgt = pair
                if src is None or tgt is None:
                    raise ValueError(f"Parallel files differ in length at line {line_no}")
                example = self._encode_pair(src, tgt, line_no)
                if self.shuffle_buffer <= 0:
                    yield example
                else:
                    buffer.append(example)
                    if len(buffer) >= self.shuffle_buffer:
                        index = rng.randrange(len(buffer))
                        yield buffer.pop(index)
            while buffer:
                yield buffer.pop(rng.randrange(len(buffer)))


def collate_batch(batch):
    srcs, tgts = zip(*batch)
    src_len = max(map(len, srcs))
    tgt_len = max(map(len, tgts))
    src_batch = torch.full((len(batch), src_len), PAD_ID, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_len), PAD_ID, dtype=torch.long)
    for i, (src, tgt) in enumerate(batch):
        src_batch[i, :len(src)] = torch.tensor(src)
        tgt_batch[i, :len(tgt)] = torch.tensor(tgt)
    return src_batch, tgt_batch


def train_sentencepiece(train_en, train_hi, model_prefix, vocab_size):
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_file = Path(str(model_prefix) + ".model")
    if not model_file.exists():
        spm.SentencePieceTrainer.train(
            input=[str(train_en), str(train_hi)],
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=0.9995,
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=0,
            hard_vocab_limit=False,
            num_threads=4,
        )
    return model_file


def run_epoch(model, loader, optimizer, criterion, device, scaler=None, training=True):
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    use_amp = device.type == "cuda"

    for source, target in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        decoder_input = target[:, :-1]
        labels = target[:, 1:]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(source, decoder_input)
                loss_sum = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss_sum).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        token_count = labels.ne(PAD_ID).sum().item()
        total_loss += loss_sum.item()
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train GRU/LSTM-attention or Transformer NMT with MLflow.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-en", default="data/processed/samanantar/en-hi/word/train.en")
    parser.add_argument("--train-hi", default="data/processed/samanantar/en-hi/word/train.hi")
    parser.add_argument("--valid-en", default="data/processed/samanantar/en-hi/word/validation.en")
    parser.add_argument("--valid-hi", default="data/processed/samanantar/en-hi/word/validation.hi")
    parser.add_argument("--architecture", choices=["gru", "lstm", "transformer"], default="transformer")
    parser.add_argument("--direction", choices=["en-hi", "hi-en"], default="en-hi")
    parser.add_argument("--attention", choices=["bahdanau", "luong"], default="bahdanau")
    parser.add_argument("--output-dir", default="artifacts/nmt")
    parser.add_argument("--spm-model", default="")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--max-train-pairs", type=int, default=50000)
    parser.add_argument("--max-valid-pairs", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.project_root).resolve()
    train_en = root / args.train_en
    train_hi = root / args.train_hi
    valid_en = root / args.valid_en
    valid_hi = root / args.valid_hi
    out = root / args.output_dir / args.architecture / args.direction
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = root / args.output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if args.spm_model:
        spm_model = Path(args.spm_model)
        if not spm_model.is_absolute():
            spm_model = root / spm_model
        if not spm_model.is_file():
            raise FileNotFoundError(spm_model)
    else:
        spm_model = train_sentencepiece(
            train_en, train_hi, tokenizer_dir / "samanantar_joint_bpe", args.vocab_size
        )
    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    vocab_size = processor.get_piece_size()

    train_data = ParallelStreamDataset(
        train_en, train_hi, processor,
        max_pairs=args.max_train_pairs,
        max_length=args.max_length,
        shuffle_buffer=5000,
        seed=args.seed,
    )
    valid_data = ParallelStreamDataset(
        valid_en, valid_hi, processor,
        max_pairs=args.max_valid_pairs,
        max_length=args.max_length,
        shuffle_buffer=0,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        collate_fn=collate_batch, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size,
        collate_fn=collate_batch, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    config = {
        "architecture": args.architecture,
        "vocab_size": vocab_size,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "max_length": args.max_length,
        "dropout": 0.1,
        "attention_type": args.attention,
    }
    if args.architecture == "transformer":
        config.update({
            "d_model": args.d_model,
            "nhead": 4,
            "num_layers": args.layers,
            "dim_feedforward": args.d_model * 4,
        })
    else:
        config.update({
            "hidden_size": args.hidden_size,
            "embedding_dim": args.hidden_size,
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_nmt_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    checkpoint_path = out / "best_model.pt"
    best_validation_loss = float("inf")

    with start_experiment_run(
        "07-neural-mt", f"nmt-{args.architecture}-samanantar",
        params={
            **config,
            "direction": args.direction,
            "training_pairs": len(train_data),
            "validation_pairs": len(valid_data),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "tokenizer": "joint SentencePiece BPE",
            "dataset_scope": "subset" if args.max_train_pairs else "full",
        },
        tags={
            "research_area": "neural-mt",
            "architecture": args.architecture,
            "direction": args.direction,
            "dataset_scope": "subset" if args.max_train_pairs else "full",
        },
    ) as run:
        log_dataset_info("train", "Samanantar", "train", len(train_data), train_en.parent)
        log_dataset_info("validation", "Samanantar", "validation", len(valid_data), valid_en.parent)
        mlflow.log_artifact(str(spm_model))

        for epoch in range(1, args.epochs + 1):
            train_data.epoch = epoch
            epoch_start = time.perf_counter()
            train_loss = run_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, training=True,
            )
            validation_loss = run_epoch(
                model, valid_loader, optimizer, criterion,
                device, scaler=None, training=False,
            )
            elapsed = time.perf_counter() - epoch_start

            mlflow.log_metrics({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch_seconds": elapsed,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss={train_loss:.4f} | "
                f"valid loss={validation_loss:.4f} | "
                f"time={elapsed/60:.1f} min",
                flush=True,
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "spm_model": str(spm_model),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                }, checkpoint_path)

        mlflow.log_metric("best_validation_loss", best_validation_loss)
        mlflow.log_artifact(str(checkpoint_path))
        config_path = out / "training_config.json"
        config_path.write_text(json.dumps({**config, **vars(args)}, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(config_path))

        mlflow.set_tags({
            "checkpoint_path": str(checkpoint_path),
            "sentencepiece_model": str(spm_model),
        })

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"SentencePiece model: {spm_model}")


if __name__ == "__main__":
    main()
