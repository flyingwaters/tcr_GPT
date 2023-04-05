import pandas as pd
import torch
import json
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 128
max_iters = 5
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1
vocab_size = 6
n_embd = 120  # head
n_head = 6
n_layer = 3
dropout = 0.1

f = pd.read_excel("./data/tcr.paire.20230321.xls")
f["tra_cdr3_nt_clean"] = f["TRA_cdr3_nt"].map(lambda x: x[4:])
f["trb_cdr3_nt_clean"] = f["TRB_cdr3_nt"].map(lambda x: x[4:])

valid = f[:1000]
train = f[1000:].reset_index(drop=True)

valid_text = "".join(
    [i+"#"+j+"$"for i, j in valid[["tra_cdr3_nt_clean", "trb_cdr3_nt_clean"]].values])

content = valid_text
chars = sorted(list(set(content)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
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

            idx_cond = idx[:,-block_size+1:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # final Time
            probs = F.softmax(logits, dim=-1)
            # sample from the prob
            idx_next = torch.max(logits, dim=1, keepdim=True)[1]
            # idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = torch.load("tcrGPT2.pt")
m = model.to(device)

right = 0
num = 1000

result = []
from tqdm import tqdm
for i, j in tqdm(valid[["tra_cdr3_nt_clean", "trb_cdr3_nt_clean"]].values):
    input_ids = torch.tensor(encode(i), dtype=torch.long).to(device)
    label = j
    
    test_input = torch.stack([input_ids])
    
    all_output = decode(m.generate(test_input, max_new_tokens=100)[0].tolist())
    output = all_output.split('$')[0]
    output_f = output.split('#')[-1]
    
    result.append({"input": i, "label": j, "prediction": output_f})
    if output_f == label:
        right += 1
df = pd.DataFrame(result)
df.to_csv("tcr_prediction.csv")
print("test! sum: 1000, right: ", right)
