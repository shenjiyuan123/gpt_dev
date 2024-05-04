import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() 
    model.train()
    return out

class MultiHeadAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # self.key = nn.Linear(n_embd, n_embd, bias=False)
        # self.query = nn.Linear(n_embd, n_embd, bias=False)
        # self.value = nn.Linear(n_embd, n_embd, bias=False)

        # check if flash attention can be used
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0") 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((1, 1, block_size, block_size))))
        self.attn_dropout = nn.Dropout(0.2)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.proj_dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=-1)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)   # (B, nh, T, hs)
        if self.flash:
            tmp_mask  = self.tril
            attn_mask = tmp_mask.masked_fill(tmp_mask==0, float('-inf'))[:, :, :T, :T]
            #  input: q, k, v need to be [B, nh, T, hs]
            #  return attn_weight @ value  -> [B, nh, T, hs]
            #  flash attn do the scale dot product only
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.2, is_causal=False)
        else:   
            wei = q @ k.transpose(2, 3) * k.size()[-1]**-0.5
            wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1) # (B, h, T, T)
            wei = self.attn_dropout(wei)
            # apply the attention weights to the values
            # (B, nh, T, T) @ (B, nh, T, hs) -> (B, h, T, hs)
            out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        out = self.proj_dropout(self.c_proj(out))
        return out
    
class FeedFoward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x
    


# super simple bigram model
class GPT2(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Block = [b1, b2, b3..]
        # python lists are not registered in a nn.Module
        self.blk = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
    

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blk(x) # (B,T,C)
        logits = self.lm_head(x)

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
            # crop the context to the last block_size tokens
            idx_conf = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_conf)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT2()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # eval process: every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
