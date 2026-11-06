from __future__ import annotations

import argparse
import json
from collections import Counter
import os
import shutil
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

import mlflow

from mtlab.experiments.tracking import start_experiment_run, log_dataset_info


def read_pairs(src_path: Path, tgt_path: Path, max_pairs: int = 0):
    """Yield parallel lines; max_pairs=0 means the complete corpus."""
    count = 0
    with src_path.open(encoding="utf-8") as src_file, tgt_path.open(encoding="utf-8") as tgt_file:
        for line_no, pair in enumerate(zip_longest(src_file, tgt_file), start=1):
            if pair[0] is None or pair[1] is None:
                raise ValueError(f"Parallel files differ in length at line {line_no}")
            src, tgt = pair
            src, tgt = src.strip(), tgt.strip()
            if not src or not tgt:
                raise ValueError(f"Empty parallel sentence at line {line_no}")
            yield src, tgt
            count += 1
            if max_pairs and count >= max_pairs:
                break


def count_a3_records(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(line.startswith("# Sentence pair") for line in f)


def require_nonempty(paths):
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty output: {path}")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def run_logged(command, cwd: Path, log_path: Path, env: dict):
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline([str(x) for x in command]) + "\n")
        log.flush()
        subprocess.run(
            [str(x) for x in command], cwd=cwd, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def monitored_run(command, cwd: Path, log_path: Path, env: dict, sample_seconds: int = 5):
    """Run an external process while recording its peak RSS and wall time."""
    import psutil

    start = time.perf_counter()
    peak_rss = 0
    last_report = start
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline([str(x) for x in command]) + "\n")
        log.flush()
        proc = subprocess.Popen(
            [str(x) for x in command], cwd=cwd, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        parent = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                try:
                    processes = [parent, *parent.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak_rss = max(peak_rss, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                now = time.perf_counter()
                if now - last_report >= 60:
                    print(
                        f"Running {Path(command[0]).name}: "
                        f"{(now-start)/3600:.2f} h elapsed; "
                        f"peak process-tree RAM {peak_rss/1024**3:.2f} GB",
                        flush=True,
                    )
                    last_report = now
                time.sleep(sample_seconds)
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
    elapsed = time.perf_counter() - start
    if code:
        raise subprocess.CalledProcessError(code, command)
    return elapsed, peak_rss


def merge_a3(prefix: Path, run_dir: Path, merge_script: Path):
    final_path = Path(str(prefix) + ".A3.final")
    parts = sorted(run_dir.glob(prefix.name + ".A3.final.part*"))
    if not parts:
        if final_path.is_file() and final_path.stat().st_size:
            return final_path
        raise FileNotFoundError(f"MGIZA produced no A3 parts/final file for {prefix}")
    if len(parts) == 1:
        shutil.copy2(parts[0], final_path)
    else:
        merge_log = run_dir / "merge.log"
        with final_path.open("w", encoding="utf-8") as out, merge_log.open("w", encoding="utf-8") as err:
            subprocess.run(
                [sys.executable, str(merge_script), *map(str, parts)],
                cwd=run_dir, stdout=out, stderr=err, check=True,
            )
    if final_path.stat().st_size == 0:
        raise RuntimeError(f"Merged A3 file is empty: {final_path}")
    return final_path



def plain2snt_python(
    source_path: Path,
    target_path: Path,
    vocab1_path: Path,
    vocab2_path: Path,
    snt1_path: Path,
    snt2_path: Path,
    log_path: Path,
) -> int:
    """Create MGIZA/GIZA-compatible VCB and SNT files in Python.

    This replaces the crashing Windows plain2snt.exe. The format follows
    MGIZA++ plain2snt.cpp:
      * vocabulary IDs 0 and 1 are reserved; lexical IDs start at 2
      * VCB rows are: ID WORD TOKEN_FREQUENCY
      * each SNT pair is: weight, source IDs, target IDs
      * reverse SNT swaps source and target IDs
    """
    source_vocab: dict[str, int] = {}
    target_vocab: dict[str, int] = {}
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    pair_count = 0
    source_path = Path(source_path)
    target_path = Path(target_path)

    with (
        source_path.open("r", encoding="utf-8-sig", newline=None) as source_file,
        target_path.open("r", encoding="utf-8-sig", newline=None) as target_file,
        Path(snt1_path).open("w", encoding="utf-8", newline="\n") as snt_forward,
        Path(snt2_path).open("w", encoding="utf-8", newline="\n") as snt_reverse,
    ):
        for line_no, pair in enumerate(
            zip_longest(source_file, target_file),
            start=1,
        ):
            source_line, target_line = pair
            if source_line is None or target_line is None:
                raise ValueError(
                    f"Parallel files differ in length at line {line_no}"
                )

            source_tokens = source_line.strip().split()
            target_tokens = target_line.strip().split()

            if not source_tokens or not target_tokens:
                raise ValueError(
                    f"Empty sentence pair at line {line_no}; "
                    "MGIZA input must not silently lose sentence IDs"
                )

            source_sequence: list[int] = []
            target_sequence: list[int] = []

            for token in source_tokens:
                if token not in source_vocab:
                    # MGIZA++ plain2snt initializes iid1 with two
                    # reserved entries, then assigns the first word ID 2.
                    source_vocab[token] = len(source_vocab) + 2
                source_counts[token] += 1
                source_sequence.append(source_vocab[token])

            for token in target_tokens:
                if token not in target_vocab:
                    target_vocab[token] = len(target_vocab) + 2
                target_counts[token] += 1
                target_sequence.append(target_vocab[token])

            # Native plain2snt writes a default sentence-pair weight of 1.0.
            snt_forward.write("1.0\n")
            snt_forward.write(" ".join(map(str, source_sequence)) + "\n")
            snt_forward.write(" ".join(map(str, target_sequence)) + "\n")

            snt_reverse.write("1.0\n")
            snt_reverse.write(" ".join(map(str, target_sequence)) + "\n")
            snt_reverse.write(" ".join(map(str, source_sequence)) + "\n")

            pair_count += 1
            if pair_count % 100_000 == 0:
                print(
                    f"Python plain2snt: {pair_count:,} pairs processed",
                    flush=True,
                )

    # Preserve first-occurrence ID assignment exactly as the native utility.
    with Path(vocab1_path).open("w", encoding="utf-8", newline="\n") as vocab_file:
        for token, token_id in sorted(source_vocab.items(), key=lambda item: item[1]):
            vocab_file.write(
                f"{token_id} {token} {source_counts[token]}\n"
            )

    with Path(vocab2_path).open("w", encoding="utf-8", newline="\n") as vocab_file:
        for token, token_id in sorted(target_vocab.items(), key=lambda item: item[1]):
            vocab_file.write(
                f"{token_id} {token} {target_counts[token]}\n"
            )

    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write("\nPython plain2snt completed.\n")
        log.write(f"Sentence pairs: {pair_count}\n")
        log.write(f"Source vocabulary size: {len(source_vocab)}\n")
        log.write(f"Target vocabulary size: {len(target_vocab)}\n")

    print(
        f"Generated MGIZA-compatible files for {pair_count:,} pairs; "
        f"vocabularies: {len(source_vocab):,} English, "
        f"{len(target_vocab):,} Hindi.",
        flush=True,
    )
    return pair_count


def prepare(args):
    project = Path(args.project_root).resolve()
    source_file = project / args.train_en
    target_file = project / args.train_hi
    mgiza_dir = Path(args.mgiza_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    mkcls = mgiza_dir / "mkcls.exe"
    mgiza = mgiza_dir / "mgiza.exe"
    merge_script = mgiza_dir / "merge_alignment.py"
    cooc_script = mgiza_dir / "snt2cooc.pl"
    coocrmp = mgiza_dir / "snt2coocrmp.exe"
    require_nonempty([source_file, target_file, mkcls, mgiza, merge_script, cooc_script, coocrmp])

    msys_usr = Path(args.msys_usr_bin)
    perl = Path(args.perl) if args.perl else msys_usr / "perl.exe"
    if not perl.is_file():
        raise FileNotFoundError(
            f"Perl not found at {perl}. Pass --perl with your MSYS2 perl.exe path."
        )

    env = os.environ.copy()
    msys_ucrt = Path(args.msys_ucrt_bin)
    env["PATH"] = os.pathsep.join([
        str(mgiza_dir), str(msys_ucrt), str(msys_usr), env.get("PATH", "")
    ])

    data_dir = work / "data"
    data_dir.mkdir(exist_ok=True)
    train_src = data_dir / "train.en"
    train_tgt = data_dir / "train.hi"

    pair_count = 0
    with train_src.open("w", encoding="utf-8", newline="\n") as src_out, train_tgt.open("w", encoding="utf-8", newline="\n") as tgt_out:
        for src, tgt in read_pairs(source_file, target_file, args.max_pairs):
            src_out.write(src + "\n")
            tgt_out.write(tgt + "\n")
            pair_count += 1

    print(f"Prepared {pair_count:,} sentence pairs ({'full corpus' if args.max_pairs == 0 else 'subset'}).")

    vocab_en = work / "train.en.vcb"
    vocab_hi = work / "train.hi.vcb"
    snt_en_hi = work / "train.en-hi.snt"
    snt_hi_en = work / "train.hi-en.snt"
    classes_en = Path(str(vocab_en) + ".classes")
    classes_hi = Path(str(vocab_hi) + ".classes")
    cooc_en_hi = work / "train.en-hi.cooc"
    cooc_hi_en = work / "train.hi-en.cooc"
    prep_log = work / "preprocessing.log"
    prep_log.write_text("", encoding="utf-8")

    with start_experiment_run(
        "01-word-alignment", "mgiza-data-preparation",
        params={
            "engine": "MGIZA++", "dataset": "Samanantar",
            "dataset_scope": "full" if args.max_pairs == 0 else "subset",
            "training_pairs": pair_count, "max_pairs": args.max_pairs,
        },
        tags={"research_area": "alignment", "stage": "preprocessing"},
    ):
        log_dataset_info("train", "Samanantar", "train", pair_count, source_file.parent)
        started = time.perf_counter()
        generated_pairs = plain2snt_python(
            source_path=train_src,
            target_path=train_tgt,
            vocab1_path=vocab_en,
            vocab2_path=vocab_hi,
            snt1_path=snt_en_hi,
            snt2_path=snt_hi_en,
            log_path=prep_log,
        )
        if generated_pairs != pair_count:
            raise ValueError(
                f"Python plain2snt generated {generated_pairs} pairs; "
                f"expected {pair_count}"
            )
        require_nonempty([vocab_en, vocab_hi, snt_en_hi, snt_hi_en])

        if sum(1 for _ in snt_en_hi.open(encoding="utf-8")) // 3 != pair_count:
            raise ValueError("Forward SNT pair count differs from prepared corpus")
        if sum(1 for _ in snt_hi_en.open(encoding="utf-8")) // 3 != pair_count:
            raise ValueError("Reverse SNT pair count differs from prepared corpus")

        run_logged([
            mkcls, "-m2", "-n10", "-c50", f"-p{train_src}", f"-V{classes_en}", "opt"
        ], work, prep_log, env)
        run_logged([
            mkcls, "-m2", "-n10", "-c50", f"-p{train_tgt}", f"-V{classes_hi}", "opt"
        ], work, prep_log, env)
        require_nonempty([classes_en, classes_hi])

        run_logged([
            perl, cooc_script, "-sort-buffer-size", args.sort_buffer,
            cooc_en_hi, vocab_en, vocab_hi, snt_en_hi,
        ], work, prep_log, env)
        run_logged([
            perl, cooc_script, "-sort-buffer-size", args.sort_buffer,
            cooc_hi_en, vocab_hi, vocab_en, snt_hi_en,
        ], work, prep_log, env)
        require_nonempty([cooc_en_hi, cooc_hi_en])
        elapsed = time.perf_counter() - started
        metadata = {
            "training_pairs": pair_count,
            "scope": "full" if args.max_pairs == 0 else "subset",
            "train_en": str(train_src), "train_hi": str(train_tgt),
            "vocab_en": str(vocab_en), "vocab_hi": str(vocab_hi),
            "snt_en_hi": str(snt_en_hi), "snt_hi_en": str(snt_hi_en),
            "cooc_en_hi": str(cooc_en_hi), "cooc_hi_en": str(cooc_hi_en),
            "classes_en": str(classes_en), "classes_hi": str(classes_hi),
        }
        metadata_path = work / "alignment_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_metrics({
            "preprocessing_seconds": elapsed,
            "training_pairs": pair_count,
            "english_vocabulary_size": count_lines(vocab_en),
            "hindi_vocabulary_size": count_lines(vocab_hi),
        })
        mlflow.log_artifact(str(prep_log))
        mlflow.log_artifact(str(metadata_path))

    return metadata, mgiza, merge_script, env


def train_one(model: str, direction: str, metadata: dict, mgiza_dir: Path,
              merge_script: Path, env: dict, work: Path, threads: int):
    src_lang, tgt_lang = direction.split("-")
    source_vocab = Path(metadata[f"vocab_{src_lang}"])
    target_vocab = Path(metadata[f"vocab_{tgt_lang}"])
    source_classes = Path(metadata[f"classes_{src_lang}"])
    target_classes = Path(metadata[f"classes_{tgt_lang}"])
    snt = Path(metadata[f"snt_{src_lang}_{tgt_lang}"])
    cooc = Path(metadata[f"cooc_{src_lang}_{tgt_lang}"])

    iteration_configs = {
        "ibm1": {"Model1_Iterations": 5, "Model2_Iterations": 0, "HMM_Iterations": 0, "Model3_Iterations": 0, "Model4_Iterations": 0, "Model5_Iterations": 0},
        "ibm2": {"Model1_Iterations": 5, "Model2_Iterations": 5, "HMM_Iterations": 0, "Model3_Iterations": 0, "Model4_Iterations": 0, "Model5_Iterations": 0},
        "hmm":  {"Model1_Iterations": 5, "Model2_Iterations": 0, "HMM_Iterations": 5, "Model3_Iterations": 0, "Model4_Iterations": 0, "Model5_Iterations": 0},
        "ibm3": {"Model1_Iterations": 5, "Model2_Iterations": 0, "HMM_Iterations": 5, "Model3_Iterations": 5, "Model4_Iterations": 0, "Model5_Iterations": 0},
        "ibm4": {"Model1_Iterations": 5, "Model2_Iterations": 0, "HMM_Iterations": 5, "Model3_Iterations": 5, "Model4_Iterations": 5, "Model5_Iterations": 0},
        "ibm5": {"Model1_Iterations": 5, "Model2_Iterations": 0, "HMM_Iterations": 5, "Model3_Iterations": 5, "Model4_Iterations": 5, "Model5_Iterations": 5},
    }
    iterations = iteration_configs[model]
    run_name = f"mgiza-{model}-{direction}-{metadata['scope']}"
    with start_experiment_run(
        "01-word-alignment", run_name,
        params={
            "engine": "MGIZA++", "model": model.upper(), "direction": direction,
            "dataset": "Samanantar", "dataset_scope": metadata["scope"],
            "training_pairs": metadata["training_pairs"], "threads": threads,
            **{k.lower(): v for k, v in iterations.items()},
        },
        tags={
            "research_area": "alignment", "engine": "mgiza",
            "model": model, "direction": direction,
            "dataset_scope": metadata["scope"],
        },
    ) as run:
        run_dir = work / "alignments" / model / direction / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        prefix = run_dir / f"{model}.{direction}"
        config = run_dir / f"{model}.{direction}.conf"
        stdout_log = run_dir / "mgiza.stdout.log"
        engine_log = Path(str(prefix) + ".engine.log")
        a3_final = Path(str(prefix) + ".A3.final")

        conf_lines = [
            f"S: {source_vocab.as_posix()}",
            f"T: {target_vocab.as_posix()}",
            f"C: {snt.as_posix()}",
            f"CoocurrenceFile: {cooc.as_posix()}",
            f"o: {prefix.as_posix()}",
            f"l: {engine_log.as_posix()}",
            "Log: 1",
            f"NCPUS: {threads}",
        ] + [f"{key}: {value}" for key, value in iterations.items()]
        config.write_text("\n".join(conf_lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(config))

        elapsed, peak_rss = monitored_run(
            [mgiza_dir / "mgiza.exe", config], run_dir, stdout_log, env
        )
        merged = merge_a3(prefix, run_dir, merge_script)
        actual = count_a3_records(merged)
        if actual != metadata["training_pairs"]:
            raise ValueError(f"Merged A3 has {actual} records; expected {metadata['training_pairs']}")

        mlflow.log_metrics({
            "training_seconds": elapsed,
            "peak_process_ram_gb": peak_rss / 1024**3,
            "training_pairs_per_second": metadata["training_pairs"] / max(elapsed, 1),
        })
        mlflow.set_tags({
            "a3_path": str(merged),
            "run_output_dir": str(run_dir),
        })
        mlflow.log_artifact(str(stdout_log))
        if engine_log.exists():
            mlflow.log_artifact(str(engine_log))
        for suffix in [".t3.final", ".a3.final", ".d3.final", ".D4.final", ".n3.final", ".p0_3.final", ".perp", ".gizacfg"]:
            parameter_file = Path(str(prefix) + suffix)
            if parameter_file.is_file():
                mlflow.log_artifact(str(parameter_file))
        merge_log = run_dir / "merge.log"
        if merge_log.exists():
            mlflow.log_artifact(str(merge_log))
        preview = run_dir / "a3_preview.txt"
        with merged.open(encoding="utf-8") as src, preview.open("w", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)
                if line.startswith("# Sentence pair") and dst.tell() > 8000:
                    break
        mlflow.log_artifact(str(preview))
        print(f"Completed {run_name}: {merged}", flush=True)
        return {
            "model": model, "direction": direction,
            "run_id": run.info.run_id,
            "a3_path": str(merged), "run_dir": str(run_dir),
        }


def main():
    parser = argparse.ArgumentParser(description="Train IBM4/IBM5 with installed MGIZA++ and MLflow.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-en", default="data/processed/samanantar/en-hi/word/train.en")
    parser.add_argument("--train-hi", default="data/processed/samanantar/en-hi/word/train.hi")
    parser.add_argument("--mgiza-dir", default="D:/mt-tools/mgiza/install")
    parser.add_argument("--work-dir", default="D:/mtlab_work/python_pipeline")
    parser.add_argument("--msys-usr-bin", default="C:/msys64/usr/bin")
    parser.add_argument("--msys-ucrt-bin", default="C:/msys64/ucrt64/bin")
    parser.add_argument("--perl", default="")
    parser.add_argument("--max-pairs", type=int, default=50000,
                        help="0 means full corpus; 50000 is a practical initial demo subset.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sort-buffer", default="1G")
    parser.add_argument("--models", default="ibm4,ibm5",
                        help="Comma-separated: ibm1,ibm2,hmm,ibm3,ibm4,ibm5")
    args = parser.parse_args()
    if args.max_pairs < 0 or args.threads < 1:
        parser.error("--max-pairs must be >= 0 and --threads must be >= 1")

    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    supported_models = {"ibm1", "ibm2", "hmm", "ibm3", "ibm4", "ibm5"}
    unsupported = sorted(set(models) - supported_models)
    if not models or unsupported:
        parser.error(
            f"Unsupported/empty model list: {unsupported or args.models}. "
            f"Choose from {sorted(supported_models)}"
        )

    work = Path(args.work_dir)
    metadata, mgiza_dir, merge_script, env = prepare(args)
    manifest = {"metadata": metadata, "runs": []}
    for model in models:
        for direction in ("en-hi", "hi-en"):
            result = train_one(
                model, direction, metadata, mgiza_dir,
                merge_script, env, work, args.threads,
            )
            manifest["runs"].append(result)
    manifest_path = work / "alignment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nAlignment training complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
