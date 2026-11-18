
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()

        self.key = nn.Linear(config["n_embd"], head_size, bias=False)
        self.query = nn.Linear(config["n_embd"], head_size, bias=False)
        self.value = nn.Linear(config["n_embd"], head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(
                config["block_size"],
                config["block_size"]
            )),
            persistent=False
        )

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        batch, time, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:time, :time] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        head_size = config["n_embd"] // config["n_head"]

        self.heads = nn.ModuleList([
            Head(head_size, config)
            for _ in range(config["n_head"])
        ])

        self.projection = nn.Linear(
            config["n_embd"],
            config["n_embd"]
        )

        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.ReLU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["dropout"])
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.ln2 = nn.LayerNorm(config["n_embd"])

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class CodeAI(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            vocab_size,
            config["n_embd"]
        )

        self.position_embedding = nn.Embedding(
            config["block_size"],
            config["n_embd"]
        )

        self.blocks = nn.Sequential(*[
            Block(config)
            for _ in range(config["n_layer"])
        ])

        self.final_norm = nn.LayerNorm(config["n_embd"])
        self.language_head = nn.Linear(
            config["n_embd"],
            vocab_size
        )

    def forward(self, idx, targets=None):
        batch, time = idx.shape

        positions = torch.arange(
            time,
            device=idx.device
        )

        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.language_head(x)
        loss = None

        if targets is not None:
            batch, time, channels = logits.shape

            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        self.eval()

        for _ in range(max_new_tokens):
            idx_conditioned = idx[:, -self.config["block_size"]:]

            logits, loss = self(idx_conditioned)

            logits = logits[:, -1, :]
            logits = logits / 0.8

            values, indices = torch.topk(
                logits,
                min(20, logits.shape[-1])
            )

            probabilities = F.softmax(values, dim=-1)

            next_token = indices.gather(
                -1,
                torch.multinomial(probabilities, 1)
            )

            idx = torch.cat((idx, next_token), dim=1)

        return idx
