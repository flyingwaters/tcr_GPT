import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import json

writer = SummaryWriter("runs_log/pretrain")

batch_size = 128
block_size = 128
max_iters = 300000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
vocab_size = 6
n_embd = 60  # 
n_head = 6
n_layer = 1
dropout = 0.0

f = pd.read_excel("data/tcr.paire.20230321.xls")
f["tra_cdr3_nt_clean"] = f["TRA_cdr3_nt"].map(lambda x: x[4:])
f["trb_cdr3_nt_clean"] = f["TRB_cdr3_nt"].map(lambda x: x[4:])

valid = f[:1000]
train = f[1000:].reset_index(drop=True)
train_text = "".join(
    [i+"#"+j+"$"for i, j in train[["tra_cdr3_nt_clean", "trb_cdr3_nt_clean"]].values])

valid_text = "".join(
    [i+"#"+j+"$"for i, j in valid[["tra_cdr3_nt_clean", "trb_cdr3_nt_clean"]].values])

content = train_text
chars = sorted(list(set(content)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

with open("stoi.json","w") as f:
    json.dump(stoi,f)
with open("itos.json","w") as f:
    json.dump(itos,f)
    

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


train_data = torch.tensor(encode(train_text), dtype=torch.long).to(device)
val_data = torch.tensor(encode(valid_text), dtype=torch.long).to(device)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(m):
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd).to(device)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd).to(device)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(
            torch.arange(T, device=device))
        x = tok_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            #
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # final Time
            probs = F.softmax(logits, dim=-1)
            # sample from the prob
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-2)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.8, patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=300, min_lr=1e-5, eps=1e-7)


for iters in range(max_iters):

    if iters % eval_interval == 0:
        losses = estimate_loss(m)
        print(
            f"step {iters}: train loss {losses['train']:.4f}. val loss {losses['val']:.4f}")
        writer.add_scalar("train_val_loss",
                          losses["train"], iters//eval_interval)
        writer.add_scalar("val_loss", losses["val"], iters//eval_interval)
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # scheduler.step(loss)
    writer.add_scalar("train_loss",loss,iters)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iters)
writer.close()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
####
torch.save(m, "tcrGPT2.pt")
