import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import NanoGPT

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 256
batch_size = 64
n_steps = 5000
lr_peak = 1e-3
warmup_steps = 500
weight_decay = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

with open('input.txt', 'r') as f:
    text = f.read()
print('training data loaded')

chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

data = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)

def get_batch(data, batch_size, seq_len):
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    batch_input = torch.stack([data[s:s+seq_len] for s in starts])
    batch_target = torch.stack([data[s+1:s+seq_len+1] for s in starts])
    return batch_input.to(device), batch_target.to(device)

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (n_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print('training started')
for step in range(n_steps):
    inputs, targets = get_batch(data, batch_size, max_seq_len)
    logits, _, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % 25 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")

torch.save(model.state_dict(), 'nanogpt_model.pt')
print('model saved.')