import torch
import numpy as np
import matplotlib.pyplot as plt
from gptv2 import GPTLanguageModel
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

# data loading
def get_batch(train_data, val_data, split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

def split_data(data, split=0.9):
    n = int(split*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def plot_loss(history):
    train_loss = [h['train'] for h in history]
    val_loss = [h['val'] for h in history]
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.savefig('loss.png')


def train(model, epochs, train_data, val_data):
    torch.cuda.empty_cache()
    history = []
    val_loss_min = np.Inf
    model_file_name = 'model.pt'

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(epochs):
        # evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Early stopping
            if losses['val'] <= val_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, losses['val']))
                torch.save(model.state_dict(), model_file_name)
                val_loss_min = losses['val']
            
            history.append(losses)

        # sample a batch of data
        xb, yb = get_batch(train_data, val_data,'train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return history

def main():
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
    train_data, val_data = split_data(data, 0.9)


    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, dropout, block_size, device).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    history = train(model, max_iters, train_data, val_data)    

    plot_loss(history)

    # context as input from the user
    context = input('Enter a context: ')
    context = torch.tensor(encode(context), dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    decoded = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(decoded)
    open('output.txt', 'w').write(decoded)

    # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    # open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))



if __name__ == "__main__":
    main()
