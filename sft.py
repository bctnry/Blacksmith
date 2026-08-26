import os
import pathlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerVocabSize
from sft_dataset import SFTDataset
from hyperparameters import d_model, n_heads, n_layers, max_seq_len

# torchrun --nproc_per_node=N --master_addr=127.0.0.1 --master_port=29500 ./sft.py

# ---------------------------------------------------------------------------
# SFT hyperparameters
# ---------------------------------------------------------------------------
batch_size = 16
n_steps = 10000
total_steps = 10000
lr_peak = 5e-5
warmup_steps = 500
weight_decay = 1e-2
ckpt_path = 'blacksmith-1_sft.pt'
pretrained_path = 'blacksmith-1_model.pt'

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
acc = torch.accelerator.current_accelerator()
backend = torch.distributed.get_default_backend_for_device(acc)
dist.init_process_group(backend)

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
device = f'cuda:{local_rank}'
is_main = (rank == 0)

if is_main:
    print(f'world_size={world_size} device={device} backend={backend}')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Add or remove JSONL files here. Each file has one conversation per line,
# either {"data": ["msg1", "msg2", ...]} (UltraChat format) or
# {"messages": [...]} format.
data_files = [
    ('identity.jsonl', 200),
    ('train_0.jsonl', 1),
]
dataset = SFTDataset(data_files, max_seq_len=max_seq_len)
if is_main:
    print(f'SFT samples: {len(dataset)}')
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                    drop_last=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
vocab_size = TokenizerVocabSize
model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)

if not pathlib.Path(pretrained_path).exists():
    raise FileNotFoundError(f'pretrained checkpoint not found: {pretrained_path}')
chkp = torch.load(pretrained_path, map_location=device)
state_dict = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
model.load_state_dict(state_dict)
if is_main:
    print(f'loaded pretrained model from {pretrained_path} (step {chkp.get("step", "?")})')

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
last_step = 0
if pathlib.Path(ckpt_path).exists():
    if is_main:
        print('SFT checkpoint found, resuming...')
    chkp = torch.load(ckpt_path, map_location=device)
    sd = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(chkp['optimizer'])
    last_step = chkp['step']

for pg in optimizer.param_groups:
    pg.setdefault('initial_lr', lr_peak)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda,
    last_epoch=-1 if last_step == 0 else last_step,
)

model = DDP(model, device_ids=[local_rank])

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if is_main:
    print('SFT started')

loss_counting = []
best_avg_loss = float('inf')
step = last_step
for epoch in range(1000):
    sampler.set_epoch(epoch)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            attn_mask = (inputs != 50256)
            logits, _, _ = model(inputs, attn_mask=attn_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if len(loss_counting) >= 50:
            loss_counting = loss_counting[1:]
        loss_counting.append(loss.item())
        avg = sum(loss_counting) / len(loss_counting)

        if is_main:
            print(f"Step {step}: loss={loss.item():.4f} avgloss={avg:.4f}")
        if is_main and step % 200 == 0:
            torch.save({
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'avg_loss': avg,
            }, ckpt_path)

        dist.barrier()

        step += 1
        if step >= total_steps:
            break
    if step >= total_steps:
        break

if is_main:
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, ckpt_path)
    print('SFT model saved.')

dist.barrier()
dist.destroy_process_group()
