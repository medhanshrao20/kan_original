"""KANInformer with paper-described and published-source execution modes.

Attention equations and embedding layout follow 375330014/lzy, commit
aebb26d9f46d2f3709ea176c51255d6955ed3707. See REPRODUCTION_AUDIT.md.
"""
import math

import torch
from kan import KAN
from torch import nn


class ProbAttention(nn.Module):
    def __init__(self, causal=False, factor=5):
        super().__init__()
        self.causal, self.factor = causal, factor

    def forward(self, q, k, v):
        b, h, length, dim = q.shape
        keys = k.shape[-2]
        samples = min(keys, max(1, self.factor * math.ceil(math.log(max(keys, 2)))))
        top = min(length, max(1, self.factor * math.ceil(math.log(max(length, 2)))))
        indices = torch.randint(keys, (length, samples), device=q.device)
        sampled = k[:, :, indices, :]
        scores = (q.unsqueeze(-2) * sampled).sum(-1)
        sparsity = scores.max(-1).values - scores.sum(-1) / keys
        selected = sparsity.topk(top, sorted=False).indices
        reduced = q.gather(2, selected.unsqueeze(-1).expand(-1, -1, -1, dim))
        scores = reduced @ k.transpose(-2, -1) / math.sqrt(dim)
        if self.causal:
            mask = torch.arange(keys, device=q.device)[None, None, None, :] > selected[..., None]
            scores = scores.masked_fill(mask, -torch.inf)
            context = v.cumsum(-2)
        else:
            context = v.mean(-2, keepdim=True).expand(-1, -1, length, -1).clone()
        update = scores.softmax(-1) @ v
        return context.scatter(2, selected.unsqueeze(-1).expand_as(update), update)


class Attention(nn.Module):
    def __init__(self, dimension, heads, factor, causal=False, full=False, mix=False):
        super().__init__()
        self.heads, self.causal, self.full = heads, causal, full
        self.mix = mix
        self.q, self.k, self.v = (nn.Linear(dimension, dimension) for _ in range(3))
        self.out = nn.Linear(dimension, dimension)
        self.prob = ProbAttention(causal, factor)

    def forward(self, query, key, value):
        def project(layer, x):
            return layer(x).reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = project(self.q, query), project(self.k, key), project(self.v, value)
        if self.full:
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            if self.causal:
                mask = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, -torch.inf)
            z = scores.softmax(-1) @ v
        else:
            z = self.prob(q, k, v)
        z = z.transpose(1, 2).contiguous()
        if self.mix:
            z = z.transpose(1, 2).contiguous()
        return self.out(z.reshape(query.shape[0], query.shape[1], -1))


class Embedding(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.value = nn.Conv1d(dimension, dimension, 3, padding=1, padding_mode="circular")
        nn.init.kaiming_normal_(self.value.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(512).float()[:, None]
        scale = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000) / dimension))
        pe = torch.zeros(512, dimension)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer("position", pe[None])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value(x.transpose(1, 2)).transpose(1, 2) + self.position[:, :x.shape[1]])


class SplineBlock(nn.Module):
    def __init__(self, dimension, length, hidden, grid, mode, device, seed):
        super().__init__()
        self.mode = mode
        width = dimension * length if mode == "author" else dimension
        self.kan = KAN(width=[width, hidden, width], grid=grid, k=3,
                       symbolic_enabled=False, save_act=False, auto_save=False,
                       device=str(device), seed=seed)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(shape[0], -1) if self.mode == "author" else x.reshape(-1, shape[-1])
        return self.kan(flat).reshape(shape)


class EncoderBlock(nn.Module):
    def __init__(self, config, length, device, seed):
        super().__init__()
        d = config.d_model
        self.attention = Attention(d, config.n_heads, config.factor)
        self.kan = SplineBlock(d, length, config.kan_hidden, config.kan_grid, config.architecture, device, seed)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attention(x, x, x)))
        return self.norm2(x + self.dropout(self.kan(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config, length, device, seed):
        super().__init__()
        d = config.d_model
        self.self_attention = Attention(d, config.n_heads, config.factor, causal=True,
                                        mix=config.architecture == "author")
        self.cross_attention = Attention(d, config.n_heads, config.factor, full=True)
        self.kan = SplineBlock(d, length, config.kan_hidden, config.kan_grid, config.architecture, device, seed)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d) for _ in range(3))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory)))
        return self.norm3(x + self.dropout(self.kan(x)))


class KANInformer(nn.Module):
    def __init__(self, features, config, device):
        super().__init__()
        self.config = config
        self.input = nn.Linear(features, config.d_model)
        self.enc_embedding, self.dec_embedding = (Embedding(config.d_model, config.dropout) for _ in range(2))
        length = config.window
        self.encoders, self.distillers = nn.ModuleList(), nn.ModuleList()
        for layer in range(config.e_layers):
            self.encoders.append(EncoderBlock(config, length, device, config.seed + layer))
            if layer < config.e_layers - 1:
                self.distillers.append(nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_model, 3, padding=1, padding_mode="circular"),
                    nn.BatchNorm1d(config.d_model), nn.ELU(), nn.MaxPool1d(3, stride=2, padding=1)))
                length = (length + 1) // 2
        decoder_length = config.window - 1 if config.architecture == "author" else config.label_len + config.horizons
        self.decoders = nn.ModuleList(DecoderBlock(config, decoder_length, device, config.seed + 100 + i)
                                      for i in range(config.d_layers))
        self.enc_norm, self.dec_norm = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, features if config.architecture == "author" else 1)

    def forward(self, x):
        if self.config.architecture == "author":
            decoder = x[:, 1:]
        else:
            zeros = x.new_zeros((len(x), self.config.horizons, x.shape[-1]))
            decoder = torch.cat([x[:, -self.config.label_len:], zeros], dim=1)
        memory = self.enc_embedding(self.input(x))
        for i, layer in enumerate(self.encoders):
            memory = layer(memory)
            if i < len(self.distillers):
                memory = self.distillers[i](memory.transpose(1, 2)).transpose(1, 2)
        memory = self.enc_norm(memory)
        z = self.dec_embedding(self.input(decoder))
        for layer in self.decoders:
            z = layer(z, memory)
        z = self.output(self.dec_norm(z))
        return z[:, -1] if self.config.architecture == "author" else z[:, -self.config.horizons:, 0]

    def forecast(self, x):
        if self.config.architecture != "author":
            return self(x)
        predictions = []
        for _ in range(self.config.horizons):
            next_features = self(x)
            predictions.append(next_features[:, -1])
            x = torch.cat([x[:, 1:], next_features[:, None]], dim=1)
        return torch.stack(predictions, dim=1)
