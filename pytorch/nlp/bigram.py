import torch
from torch import nn
from torch.nn import functional as F

#parameters
batch_size = 32
block_size = 8
max_iter = 3000
max_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iter = 200
n_embd = 32
#
torch.manual_seed(42)

with open("input.txt","r") as f:
    text = f.read()

tokens = sorted(list(set(text)))
stoi = { char:i for i,char in enumerate(tokens)}
itos = { i:char for i,char in enumerate(tokens)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join(itos[c] for c in l)

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data=train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.ln = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        
        tok_emb = self.token_embedding_table(idx)
        logits = self.ln(tok_emb)
       
        # B,T,C = logits.shape
        # logits = logits.view(B*T,C)
        # # targets= targets.view(B*T)
        # loss = F.cross_entropy(logits,targets)

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

m = BigramLanguageModel(x,y)



@torch.no_grad()
