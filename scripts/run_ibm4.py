
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

import mlflow
import psutil

from mtlab.experiments.tracking import (
    start_experiment_run,
    log_dataset_info,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data/processed/samanantar/en-hi/word"

TRAIN_EN = TRAIN_DIR / "train.en"
TRAIN_HI = TRAIN_DIR / "train.hi"

WORK_TRAIN = Path("D:/mtlab_work/samanantar_word")
WORK_DIR = Path("D:/mtlab_work/full_en_hi")

MGIZA_DIR = Path("D:/mt-tools/mgiza/install")

PLAIN2SNT = MGIZA_DIR / "plain2snt.exe"
MKCLS = MGIZA_DIR / "mkcls.exe"
MGIZA = MGIZA_DIR / "mgiza.exe"

MERGE_SCRIPT = MGIZA_DIR / "merge_alignment.py"
SNT2COOC_SCRIPT = MGIZA_DIR / "snt2cooc.pl"

MSYS_UCRT_BIN = Path("C:/msys64/ucrt64/bin")
MSYS_USR_BIN = Path("C:/msys64/usr/bin")

METADATA_PATH = WORK_DIR / "ibm4_metadata.json"


# ============================================================
# VALIDATION
# ============================================================

def validate_parallel(en_path: Path, hi_path: Path) -> int:
    """Validate parallel files without requiring sentence IDs."""

    count = 0

    with (
        open(en_path, encoding="utf-8") as en_file,
        open(hi_path, encoding="utf-8") as hi_file,
    ):
        for line_no, rows in enumerate(
            zip_longest(en_file, hi_file),
            start=1,
        ):
            if any(row is None for row in rows):
                raise ValueError(
                    f"English and Hindi files have different "
                    f"lengths at line {line_no}"
                )

            en, hi = rows

            if not en.strip() or not hi.strip():
                raise ValueError(
                    f"Empty sentence pair at line {line_no}"
                )

            count += 1

    return count


def count_snt_pairs(path: Path) -> int:
    """Count sentence pairs in the three-line SNT format."""

    with open(path, encoding="utf-8") as file:
        lines = sum(1 for _ in file)

    if lines % 3 != 0:
        raise ValueError(f"Invalid SNT file: {path}")

    return lines // 3


def count_a3_records(path: Path) -> int:
    """Count sentence records in an A3 alignment file."""

    count = 0

    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("# Sentence pair"):
                count += 1

    return count


def require_files(paths):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def require_outputs(paths):
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing or empty generated file: {path}"
            )


# ============================================================
# ENVIRONMENT
# ============================================================

def make_environment():
    env = os.environ.copy()

    extra_paths = [
        str(MGIZA_DIR),
        str(MSYS_UCRT_BIN),
        str(MSYS_USR_BIN),
    ]

    env["PATH"] = os.pathsep.join(
        extra_paths + [env.get("PATH", "")]
    )

    perl = shutil.which(
        "perl",
        path=env["PATH"],
    )

    if perl is None:
        raise RuntimeError(
            "Perl not found. Install it in MSYS2 using: pacman -S perl"
        )

    return env, perl


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(command, log_path: Path, env):
    print(
        "\nRunning:",
        subprocess.list2cmdline(command),
        flush=True,
    )

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            "\n$ " + subprocess.list2cmdline(command) + "\n"
        )
        log.flush()

        subprocess.run(
            command,
            cwd=WORK_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print("Completed.", flush=True)


# ============================================================
# FULL-CORPUS PREPROCESSING
# ============================================================

def prepare_data():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    require_files([
        TRAIN_EN,
        TRAIN_HI,
        WORK_TRAIN / "train.en",
        WORK_TRAIN / "train.hi",
        PLAIN2SNT,
        MKCLS,
        MGIZA,
        MERGE_SCRIPT,
        SNT2COOC_SCRIPT,
    ])

    # Validate original corpus.
    train_pairs = validate_parallel(
        TRAIN_EN,
        TRAIN_HI,
    )

    # Validate the workspace junction.
    workspace_pairs = validate_parallel(
        WORK_TRAIN / "train.en",
        WORK_TRAIN / "train.hi",
    )

    if train_pairs != workspace_pairs:
        raise ValueError(
            "Workspace corpus differs from the project corpus."
        )

    env, perl = make_environment()

    # Generated files.
    vocab_en = WORK_DIR / "full.en.vcb"
    vocab_hi = WORK_DIR / "full.hi.vcb"

    snt_en_hi = WORK_DIR / "full.en-hi.snt"
    snt_hi_en = WORK_DIR / "full.hi-en.snt"

    classes_en = WORK_DIR / "full.en.vcb.classes"
    classes_hi = WORK_DIR / "full.hi.vcb.classes"

    cooc_en_hi = WORK_DIR / "full.en-hi.cooc"
    cooc_hi_en = WORK_DIR / "full.hi-en.cooc"

    log_path = WORK_DIR / "preprocessing.log"
    log_path.write_text("", encoding="utf-8")

    with start_experiment_run(
        experiment_name="01-word-alignment",
        run_name="mgiza-ibm4-full-preprocessing",
        params={
            "engine": "MGIZA++",
            "model": "IBM4",
            "dataset": "Samanantar",
            "dataset_scope": "full",
            "training_pairs": train_pairs,
            "tokenizer": "Moses + Indic NLP",
        },
        tags={
            "research_area": "word-alignment",
            "engine": "mgiza",
            "model": "ibm4",
            "stage": "preprocessing",
            "dataset_scope": "full",
        },
    ):
        log_dataset_info(
            role="train",
            name="Samanantar",
            split="train",
            row_count=train_pairs,
            path=TRAIN_DIR,
        )

        started = time.perf_counter()

        # 1. Vocabulary and sentence-pair files.
        run_command(
            [
                str(PLAIN2SNT),
                str(WORK_TRAIN / "train.en"),
                str(WORK_TRAIN / "train.hi"),
                "-vcb1", str(vocab_en),
                "-vcb2", str(vocab_hi),
                "-snt1", str(snt_en_hi),
                "-snt2", str(snt_hi_en),
            ],
            log_path,
            env,
        )

        require_outputs([
            vocab_en,
            vocab_hi,
            snt_en_hi,
            snt_hi_en,
        ])

        if count_snt_pairs(snt_en_hi) != train_pairs:
            raise ValueError(
                "Forward SNT sentence count is incorrect."
            )

        if count_snt_pairs(snt_hi_en) != train_pairs:
            raise ValueError(
                "Reverse SNT sentence count is incorrect."
            )

        # 2. English word classes.
        run_command(
            [
                str(MKCLS),
                "-m2",
                "-n10",
                "-c50",
                f"-p{WORK_TRAIN / 'train.en'}",
                f"-V{classes_en}",
                "opt",
            ],
            log_path,
            env,
        )

        # 3. Hindi word classes.
        run_command(
            [
                str(MKCLS),
                "-m2",
                "-n10",
                "-c50",
                f"-p{WORK_TRAIN / 'train.hi'}",
                f"-V{classes_hi}",
                "opt",
            ],
            log_path,
            env,
        )

        require_outputs([
            classes_en,
            classes_hi,
        ])

        # 4. English -> Hindi co-occurrence.
        run_command(
            [
                perl,
                str(SNT2COOC_SCRIPT),
                "-sort-buffer-size",
                "1G",
                str(cooc_en_hi),
                str(vocab_en),
                str(vocab_hi),
                str(snt_en_hi),
            ],
            log_path,
            env,
        )

        # 5. Hindi -> English co-occurrence.
        run_command(
            [
                perl,
                str(SNT2COOC_SCRIPT),
                "-sort-buffer-size",
                "1G",
                str(cooc_hi_en),
                str(vocab_hi),
                str(vocab_en),
                str(snt_hi_en),
            ],
            log_path,
            env,
        )

        require_outputs([
            cooc_en_hi,
            cooc_hi_en,
        ])

        elapsed = time.perf_counter() - started

        metadata = {
            "training_pairs": train_pairs,
            "training_snt_en_hi": str(snt_en_hi),
            "training_snt_hi_en": str(snt_hi_en),
            "vocab_en": str(vocab_en),
            "vocab_hi": str(vocab_hi),
            "classes_en": str(classes_en),
            "classes_hi": str(classes_hi),
            "cooc_en_hi": str(cooc_en_hi),
            "cooc_hi_en": str(cooc_hi_en),
        }

        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        mlflow.log_metrics({
            "preprocessing_seconds": elapsed,
            "training_pairs": train_pairs,
        })

        mlflow.log_artifact(str(log_path))
        mlflow.log_artifact(str(METADATA_PATH))

    print("\nFull-corpus preprocessing completed.")
    print(f"Training sentence pairs: {train_pairs}")
    print(f"Preprocessing time: {elapsed / 60:.2f} minutes")

    return metadata


# ============================================================
# PROCESS MEMORY MONITORING
# ============================================================

def process_memory_bytes(process):
    total = 0

    try:
        processes = [process]
        processes.extend(process.children(recursive=True))

        for item in processes:
            try:
                total += item.memory_info().rss
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass

    return total


def run_monitored(command, cwd, log_path, env):
    started = time.perf_counter()
    peak_memory = 0
    last_report = started

    print(
        "\nStarting MGIZA++:",
        subprocess.list2cmdline(command),
        flush=True,
    )

    with open(log_path, "w", encoding="utf-8") as output:
        output.write(
            subprocess.list2cmdline(command) + "\n"
        )
        output.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

        monitored = psutil.Process(process.pid)

        try:
            while process.poll() is None:
                peak_memory = max(
                    peak_memory,
                    process_memory_bytes(monitored),
                )

                now = time.perf_counter()

                if now - last_report >= 60:
                    print(
                        f"MGIZA++ running | "
                        f"elapsed={(now-started)/3600:.2f} hours | "
                        f"peak RAM={peak_memory/(1024**3):.2f} GB",
                        flush=True,
                    )
                    last_report = now

                time.sleep(5)

            return_code = process.wait()

        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

    elapsed = time.perf_counter() - started

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )

    return elapsed, peak_memory


# ============================================================
# A3 OUTPUT MERGING
# ============================================================

def merge_training_a3(prefix: Path, run_dir: Path):
    final_path = Path(
        str(prefix) + ".A3.final"
    )

    parts = sorted(
        run_dir.glob(
            prefix.name + ".A3.final.part*"
        )
    )

    if not parts:
        if final_path.is_file() and final_path.stat().st_size > 0:
            return final_path

        raise FileNotFoundError(
            f"No A3 training alignment parts found: {prefix}"
        )

    if len(parts) == 1:
        shutil.copy2(parts[0], final_path)

    else:
        merge_log = run_dir / "merge.log"

        command = [
            sys.executable,
            str(MERGE_SCRIPT),
            *[str(path) for path in parts],
        ]

        with (
            open(final_path, "w", encoding="utf-8") as output,
            open(merge_log, "w", encoding="utf-8") as error_log,
        ):
            subprocess.run(
                command,
                cwd=run_dir,
                stdout=output,
                stderr=error_log,
                check=True,
            )

    if final_path.stat().st_size == 0:
        raise RuntimeError("Merged A3 file is empty.")

    return final_path


def create_preview(source, destination, max_sentences=20):
    written = 0

    with (
        open(source, encoding="utf-8") as input_file,
        open(destination, "w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            output_file.write(line)

            if line.startswith("# Sentence pair"):
                written += 1

            if written >= max_sentences:
                break


# ============================================================
# IBM4 TRAINING
# ============================================================

def train_direction(direction, metadata, threads, env):
    if direction == "en-hi":
        source_vocab = WORK_DIR / "full.en.vcb"
        target_vocab = WORK_DIR / "full.hi.vcb"
        training_snt = WORK_DIR / "full.en-hi.snt"
        cooc = WORK_DIR / "full.en-hi.cooc"

    elif direction == "hi-en":
        source_vocab = WORK_DIR / "full.hi.vcb"
        target_vocab = WORK_DIR / "full.en.vcb"
        training_snt = WORK_DIR / "full.hi-en.snt"
        cooc = WORK_DIR / "full.hi-en.cooc"

    else:
        raise ValueError(direction)

    run_name = f"mgiza-ibm4-{direction}-full"

    params = {
        "engine": "MGIZA++",
        "model": "IBM4",
        "direction": direction,
        "dataset": "Samanantar",
        "dataset_scope": "full",
        "training_pairs": metadata["training_pairs"],
        "threads": threads,
        "model1_iterations": 5,
        "model2_iterations": 0,
        "hmm_iterations": 5,
        "model3_iterations": 5,
        "model4_iterations": 5,
        "model5_iterations": 0,
    }

    tags = {
        "research_area": "word-alignment",
        "engine": "mgiza",
        "model": "ibm4",
        "direction": direction,
        "dataset_scope": "full",
        "stage": "training",
    }

    with start_experiment_run(
        experiment_name="01-word-alignment",
        run_name=run_name,
        params=params,
        tags=tags,
    ) as run:

        run_dir = (
            WORK_DIR
            / "runs"
            / "ibm4"
            / direction
            / run.info.run_id
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = run_dir / f"ibm4.{direction}"

        config_path = run_dir / f"ibm4.{direction}.conf"
        stdout_path = run_dir / "mgiza.stdout.log"

        engine_log = Path(
            str(prefix) + ".engine.log"
        )

        # MGIZA defaults:
        # source vocabulary classes = S + ".classes"
        # target vocabulary classes = T + ".classes"
        #
        # We generated those exact filenames during preprocessing.

        config = f"""
S: {source_vocab.as_posix()}
T: {target_vocab.as_posix()}
C: {training_snt.as_posix()}
CoocurrenceFile: {cooc.as_posix()}
o: {prefix.as_posix()}
l: {engine_log.as_posix()}
Log: 1
NCPUS: {threads}
Model1_Iterations: 5
Model2_Iterations: 0
HMM_Iterations: 5
Model3_Iterations: 5
Model4_Iterations: 5
Model5_Iterations: 0
""".strip()

        config_path.write_text(
            config + "\n",
            encoding="utf-8",
        )

        mlflow.log_artifact(str(config_path))

        mlflow.set_tags({
            "output_directory": str(run_dir),
            "output_prefix": str(prefix),
        })

        elapsed, peak_memory = run_monitored(
            [str(MGIZA), str(config_path)],
            run_dir,
            stdout_path,
            env,
        )

        # Merge multithreaded training alignments.
        train_a3 = merge_training_a3(
            prefix,
            run_dir,
        )

        expected_pairs = metadata["training_pairs"]
        actual_pairs = count_a3_records(train_a3)

        if actual_pairs != expected_pairs:
            raise ValueError(
                f"Training A3 contains {actual_pairs} records, "
                f"expected {expected_pairs}"
            )

        preview = run_dir / "training_alignment_preview.A3"

        create_preview(
            train_a3,
            preview,
        )

        mlflow.log_metrics({
            "training_seconds": elapsed,
            "peak_process_ram_gb": peak_memory / (1024 ** 3),
            "training_pairs_per_second": (
                expected_pairs / max(elapsed, 1)
            ),
        })

        mlflow.set_tags({
            "training_a3_path": str(train_a3),
        })

        mlflow.log_artifact(str(stdout_path))
        mlflow.log_artifact(str(preview))

        if engine_log.is_file():
            mlflow.log_artifact(str(engine_log))

        merge_log = run_dir / "merge.log"

        if merge_log.is_file():
            mlflow.log_artifact(str(merge_log))

        # Log model parameter tables, without uploading
        # the potentially huge A3 training file again.
        for suffix in [
            ".t3.final",
            ".d3.final",
            ".D4.final",
            ".n3.final",
            ".p0_3.final",
            ".perp",
            ".gizacfg",
        ]:
            path = Path(str(prefix) + suffix)

            if path.is_file():
                mlflow.log_artifact(str(path))

        print(f"\nCompleted: {run_name}", flush=True)
        print(f"MLflow run ID: {run.info.run_id}", flush=True)
        print(f"Training A3: {train_a3}", flush=True)
        print(
            f"Training duration: {elapsed / 3600:.2f} hours",
            flush=True,
        )
        print(
            f"Peak process RAM: {peak_memory / (1024 ** 3):.2f} GB",
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-corpus MGIZA++ IBM4 training."
    )

    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "train"],
        default="all",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("Threads must be at least 1.")

    if args.stage in ("all", "prepare"):
        metadata = prepare_data()

    else:
        if not METADATA_PATH.is_file():
            raise FileNotFoundError(
                "IBM4 preprocessing has not been completed. "
                "Run --stage prepare first."
            )

        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

    if args.stage in ("all", "train"):
        env, _ = make_environment()

        for direction in ("en-hi", "hi-en"):
            train_direction(
                direction,
                metadata,
                args.threads,
                env,
            )

    print(
        "\nIBM4 training workflow completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
