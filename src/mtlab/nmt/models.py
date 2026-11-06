from __future__ import annotations

import math

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over bidirectional encoder states."""

    def __init__(self, query_dim: int, key_dim: int, attention_dim: int):
        super().__init__()
        self.query = nn.Linear(query_dim, attention_dim, bias=False)
        self.key = nn.Linear(key_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, query, keys, padding_mask):
        # query: [batch, query_dim], keys: [batch, source_len, key_dim]
        scores = self.energy(
            torch.tanh(self.key(keys) + self.query(query).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class LuongAttention(nn.Module):
    """Luong dot-product attention over encoder states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, query, keys, padding_mask):
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) * self.scale
        scores = scores.masked_fill(padding_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class RecurrentAttentionNMT(nn.Module):
    """GRU/LSTM encoder-decoder with Bahdanau additive attention."""

    def __init__(self, vocab_size: int, pad_id: int, hidden_size: int = 128,
                 embedding_dim: int = 128, rnn_type: str = "gru",
                 attention_type: str = "bahdanau", dropout: float = 0.2):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.rnn_type = rnn_type
        attention_type = attention_type.lower()
        if attention_type not in {"bahdanau", "luong"}:
            raise ValueError("attention_type must be 'bahdanau' or 'luong'")
        self.attention_type = attention_type
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        cell_cls = nn.GRUCell if rnn_type == "gru" else nn.LSTMCell
        self.encoder = rnn_cls(
            embedding_dim, hidden_size, batch_first=True,
            bidirectional=True,
        )
        enc_dim = hidden_size * 2
        self.attention = (
            AdditiveAttention(enc_dim, enc_dim, enc_dim)
            if attention_type == "bahdanau"
            else LuongAttention(enc_dim)
        )
        self.decoder_cell = cell_cls(embedding_dim + enc_dim, enc_dim)
        self.output = nn.Linear(enc_dim * 2, vocab_size)

    def encode(self, source):
        padding_mask = source.eq(self.pad_id)
        embedded = self.dropout(self.embedding(source))
        encoder_outputs, state = self.encoder(embedded)
        if self.rnn_type == "gru":
            hidden = torch.cat([state[-2], state[-1]], dim=-1)
            decoder_state = hidden
        else:
            h, c = state
            decoder_state = (
                torch.cat([h[-2], h[-1]], dim=-1),
                torch.cat([c[-2], c[-1]], dim=-1),
            )
        return encoder_outputs, padding_mask, decoder_state

    def decode_step(self, token, state, encoder_outputs, source_padding_mask):
        embedded = self.dropout(self.embedding(token))
        if self.rnn_type == "gru":
            query = state
        else:
            query = state[0]
        context, attention = self.attention(
            query, encoder_outputs, source_padding_mask
        )
        decoder_input = torch.cat([embedded, context], dim=-1)
        new_state = self.decoder_cell(decoder_input, state)
        hidden = new_state if self.rnn_type == "gru" else new_state[0]
        logits = self.output(torch.cat([hidden, context], dim=-1))
        return logits, new_state, attention

    def forward(self, source, target_input):
        encoder_outputs, source_padding_mask, state = self.encode(source)
        logits = []
        for position in range(target_input.size(1)):
            step_logits, state, _ = self.decode_step(
                target_input[:, position], state,
                encoder_outputs, source_padding_mask,
            )
            logits.append(step_logits.unsqueeze(1))
        return torch.cat(logits, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerNMT(nn.Module):
    """Encoder-decoder Transformer for subword machine translation."""

    def __init__(self, vocab_size: int, pad_id: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_length: int = 512):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_length, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, source, target_input):
        source_padding_mask = source.eq(self.pad_id)
        target_padding_mask = target_input.eq(self.pad_id)
        target_length = target_input.size(1)
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, device=target_input.device, dtype=torch.bool),
            diagonal=1,
        )
        src = self.position(self.embedding(source) * math.sqrt(self.d_model))
        tgt = self.position(self.embedding(target_input) * math.sqrt(self.d_model))
        hidden = self.transformer(
            src, tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(hidden)


def build_nmt_model(config: dict) -> nn.Module:
    architecture = config["architecture"].lower()
    if architecture in {"gru", "lstm"}:
        return RecurrentAttentionNMT(
            vocab_size=config["vocab_size"],
            pad_id=config["pad_id"],
            hidden_size=config.get("hidden_size", 128),
            embedding_dim=config.get("embedding_dim", 128),
            rnn_type=architecture,
            attention_type=config.get("attention_type", "bahdanau"),
            dropout=config.get("dropout", 0.2),
        )
    if architecture == "transformer":
        return TransformerNMT(
            vocab_size=config["vocab_size"],
            pad_id=config["pad_id"],
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
            max_length=config.get("max_length", 256),
        )
    raise ValueError(f"Unsupported NMT architecture: {architecture}")


@torch.no_grad()
def greedy_decode(model, source, bos_id: int, eos_id: int,
                  max_length: int = 128, pad_id: int = 3):
    """Greedy decoding for recurrent-attention and Transformer models."""
    model.eval()
    batch_size = source.size(0)
    device = source.device
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if isinstance(model, RecurrentAttentionNMT):
        encoder_outputs, source_padding_mask, state = model.encode(source)
        for _ in range(max_length):
            logits, state, _ = model.decode_step(
                generated[:, -1], state, encoder_outputs, source_padding_mask
            )
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if finished.all():
                break
    else:
        for _ in range(max_length):
            logits = model(source, generated)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if finished.all():
                break
    return generated
