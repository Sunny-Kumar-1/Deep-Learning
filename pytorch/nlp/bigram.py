from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path

#parameters
batch_size = 32
block_size = 8
max_iter = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iter = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#
torch.manual_seed(42)
path = Path("pytorch/nlp")
with open(path/"input.txt","r") as f:
    text = f.read()

#encoder & decoder
tokens = sorted(list(set(text)))
vocab_size = len(tokens)
stoi = { char:i for i,char in enumerate(tokens)}
itos = { i:char for i,char in enumerate(tokens)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join(itos[c] for c in l)

#split data train & test
data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data=train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

@torch.no_grad()
def estimate_loss(model:torch.nn.Module):
    out={}
    model.eval()
    with torch.inference_mode():
        for split in ["train","val"]:
            losses = torch.zeros(eval_iter)
            for k in range(eval_iter):
                X,Y = get_batch(split)
                logits,loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        # self.token_embedding_table2 = nn.Embedding(n_embd,n_embd)
        self.ln = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        # pos_emb = self.token_embedding_table2(torch.arange(T,device=device))

        # x = tok_emb +pos_emb

        logits = self.ln(tok_emb)


        if targets is None :
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits,loss

    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            logits ,loss = self(idx)

            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(probs,num_samples=1)

            idx = torch.cat((idx,idx_next),dim=1)

        return idx

# m = BigramLanguageModel(x,y)

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C =x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) *C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float("-inf"))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        out = wei@v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx

# model = BigramLanguageModel(vocab_size)
model = GPTLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

def train(model:torch.nn.Module,
          optimizer:torch.optim.Optimizer):

    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step{iter}:train loss {losses["train"]:.4f},val loss {losses["val"]:.4f}")

        xb , yb = get_batch("train")

        logits , loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

train(model=m,optimizer=optimizer)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_token=500)[0].tolist()))