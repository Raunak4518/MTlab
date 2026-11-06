from pathlib import Path
import tempfile

from mtlab.smt.phrase_based import (
    iter_a3, grow_diag_final_and, train_phrase_table,
    train_language_model, PhraseBasedDecoder,
)


def test_phrase_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        en = root / 'train.en'
        hi = root / 'train.hi'
        en.write_text('the house\nthe book\n', encoding='utf-8')
        hi.write_text('das haus\ndas buch\n', encoding='utf-8')

        fwd = root / 'fwd.A3'
        rev = root / 'rev.A3'
        fwd.write_text(
            '# Sentence pair (1) source length 2 target length 2 alignment score : 0.1\n'
            'the house\n'
            'das ({ 1 }) haus ({ 2 })\n'
            '# Sentence pair (2) source length 2 target length 2 alignment score : 0.1\n'
            'the book\n'
            'das ({ 1 }) buch ({ 2 })\n', encoding='utf-8')
        rev.write_text(
            '# Sentence pair (1) source length 2 target length 2 alignment score : 0.1\n'
            'das haus\n'
            'the ({ 1 }) house ({ 2 })\n'
            '# Sentence pair (2) source length 2 target length 2 alignment score : 0.1\n'
            'das buch\n'
            'the ({ 1 }) book ({ 2 })\n', encoding='utf-8')

        db = root / 'phrases.sqlite'
        stats = train_phrase_table(en, hi, fwd, rev, db, max_phrase_len=3)
        assert stats['sentence_pairs'] == 2
        assert stats['phrase_pairs'] >= 4

        reverse_stats = train_phrase_table(
            hi, en, rev, fwd, root / 'phrases-reverse.sqlite', max_phrase_len=3
        )
        assert reverse_stats['sentence_pairs'] == 2
        assert reverse_stats['phrase_pairs'] >= 4

        lm = train_language_model(hi, order=3)
        decoder = PhraseBasedDecoder(db, lm, beam_size=8)
        hyp = decoder.decode(['the', 'house'])
        decoder.close()
        assert isinstance(hyp, str) and hyp
        assert 'das' in hyp


if __name__ == '__main__':
    test_phrase_pipeline()
    print('phrase pipeline passed')
