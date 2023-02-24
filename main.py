import torch
from gpt import GPT
from preprocesssing import preprocess

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

if __name__ == "__main__":
    torch.manual_seed(1337)

    # run preprocess.py to get jokes.txt
    preprocess()

    with open('jokes.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # character encoding decoding
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
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
        return out

    model = GPT(vocab_size, n_embd, n_head, n_layer, dropout, block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(max_iters):
        model.train()
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f'Iteration {i} | Train loss {losses["train"]:.3f} | Val loss {losses["val"]:.3f}')
        

    # generate some jokes
    model.eval()
    for _ in range(10):
        x = torch.randint(vocab_size, (1,1)).to(device)
        x = model.generate(x, max_len=500)
        print(decode(x[0].tolist()))
