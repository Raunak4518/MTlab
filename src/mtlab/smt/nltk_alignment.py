from __future__ import annotations

from collections import Counter
from pathlib import Path

from nltk.translate import AlignedSent
from nltk.translate.ibm4 import IBMModel4
from nltk.translate.ibm5 import IBMModel5


def frequency_rank_classes(sentences, n_classes: int = 50) -> dict[str, int]:
    """Assign deterministic frequency-rank word classes for NLTK IBM4/5.

    This is a lightweight statistical classing approximation, not Brown's
    mkcls clustering. Use MGIZA++/mkcls for the full-scale main experiment.
    """
    counts = Counter(token for sentence in sentences for token in sentence)
    words = sorted(counts, key=lambda word: (-counts[word], word))
    if not words:
        return {}
    return {
        word: min(n_classes - 1, rank * n_classes // len(words))
        for rank, word in enumerate(words)
    }


def train_nltk_ibm(
    source_path: str | Path,
    target_path: str | Path,
    model_name: str,
    iterations: int = 1,
    max_pairs: int = 500,
):
    """Train NLTK IBM4 or IBM5 on a small corpus and return model + alignments.

    AlignedSent.words is target; AlignedSent.mots is source.
    Returned alignments use Pharaoh source-target zero-based coordinates.
    """
    source_sentences = Path(source_path).read_text(encoding="utf-8").splitlines()
    target_sentences = Path(target_path).read_text(encoding="utf-8").splitlines()
    if len(source_sentences) != len(target_sentences):
        raise ValueError("Parallel source and target files have different line counts")
    if max_pairs:
        source_sentences = source_sentences[:max_pairs]
        target_sentences = target_sentences[:max_pairs]
    if not source_sentences:
        raise ValueError("No training pairs")
    for index, (src, tgt) in enumerate(zip(source_sentences, target_sentences), start=1):
        if not src.strip() or not tgt.strip():
            raise ValueError(f"Empty parallel sentence at line {index}")

    source_tokens = [line.split() for line in source_sentences]
    target_tokens = [line.split() for line in target_sentences]
    corpus = [
        AlignedSent(words=tgt, mots=src)
        for src, tgt in zip(source_tokens, target_tokens)
    ]
    source_classes = frequency_rank_classes(source_tokens)
    target_classes = frequency_rank_classes(target_tokens)

    model_name = model_name.lower()
    if model_name == "ibm4":
        model = IBMModel4(corpus, iterations, source_classes, target_classes)
    elif model_name == "ibm5":
        model = IBMModel5(corpus, iterations, source_classes, target_classes)
    else:
        raise ValueError("model_name must be ibm4 or ibm5")

    alignments = []
    for pair in corpus:
        # Alignment tuples are (target_index, source_index) in NLTK.
        alignments.append({(src, tgt) for tgt, src in pair.alignment})
    return model, alignments, source_sentences, target_sentences
