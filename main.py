import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import NanoGPT
from dataset2 import DatasetManager, Tokenizer

d_model = 768
n_heads = 12
n_layers = 12
max_seq_len = 1024
batch_size = 32
n_steps = 5000
total_steps = 1000000
lr_peak = 2.5e-4
warmup_steps = 5000
weight_decay = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

dataset = DatasetManager(device, 'tokens.bin')
print('training data loaded')
vocab_size = Tokenizer.n_vocab

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print('training started')

last_step = 0
if pathlib.Path('blacksmith-1_model.pt').exists():
    print('previous checkpoint found. loading....')
    chkp = torch.load('blacksmith-1_model.pt')
    model.load_state_dict(chkp['model'])
    optimizer.load_state_dict(chkp['optimizer'])
    last_step = chkp['step']
    
for step in range(last_step, min(last_step+n_steps, total_steps)):
    inputs, targets = dataset.get_batch(batch_size, max_seq_len)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    print(f"Step {step}: loss={loss.item():.6f}")
    if step % 50 == 0:
        print('saving checkpoint...')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, 'blacksmith-1_model.pt')

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': step,
}, 'blacksmith-1_model.pt')
print('model saved.')

