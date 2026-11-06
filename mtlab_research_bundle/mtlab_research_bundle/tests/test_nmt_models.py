import torch
from mtlab.nmt.models import build_nmt_model, greedy_decode


def test_nmt_architectures():
    torch.manual_seed(7)
    batch, src_len, tgt_len, vocab = 3, 6, 5, 40
    source = torch.randint(4, vocab, (batch, src_len))
    target = torch.randint(4, vocab, (batch, tgt_len))
    source[0, -1] = 3
    target[0, -1] = 3

    configs = [
        {"architecture": "gru", "vocab_size": vocab, "pad_id": 3, "hidden_size": 16, "embedding_dim": 16},
        {"architecture": "lstm", "vocab_size": vocab, "pad_id": 3, "hidden_size": 16, "embedding_dim": 16, "attention_type": "bahdanau"},
        {"architecture": "gru", "vocab_size": vocab, "pad_id": 3, "hidden_size": 16, "embedding_dim": 16, "attention_type": "luong"},
        {"architecture": "transformer", "vocab_size": vocab, "pad_id": 3, "d_model": 32, "nhead": 4, "num_layers": 1, "dim_feedforward": 64, "max_length": 32},
    ]
    for config in configs:
        model = build_nmt_model(config)
        logits = model(source, target[:, :-1])
        assert logits.shape == (batch, tgt_len - 1, vocab)
        generated = greedy_decode(model, source, bos_id=1, eos_id=2, max_length=5, pad_id=3)
        assert generated.shape[0] == batch
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), target[:, 1:].reshape(-1), ignore_index=3
        )
        loss.backward()


if __name__ == "__main__":
    test_nmt_architectures()
    print("NMT architectures passed")
