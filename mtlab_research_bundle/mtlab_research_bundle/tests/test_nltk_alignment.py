from pathlib import Path
import tempfile

from mtlab.smt.nltk_alignment import train_nltk_ibm


def test_nltk_ibm4_ibm5():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "train.en"
        target = root / "train.hi"
        source.write_text(
            "the house\nthe book\na house\na book\nthe small house\nthe big book\n",
            encoding="utf-8",
        )
        target.write_text(
            "das haus\ndas buch\nein haus\nein buch\ndas kleine haus\ndas große buch\n",
            encoding="utf-8",
        )
        for model_name in ("ibm4", "ibm5"):
            model, alignments, src, tgt = train_nltk_ibm(
                source, target, model_name, iterations=1, max_pairs=6
            )
            assert len(alignments) == len(src) == len(tgt) == 6
            assert alignments[0]


if __name__ == "__main__":
    test_nltk_ibm4_ibm5()
    print("NLTK IBM4/IBM5 smoke test passed")
