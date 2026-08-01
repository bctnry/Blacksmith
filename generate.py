import torch
import torch.nn.functional as F
from model import NanoGPT

d_model = 128
n_heads = 4
n_layers = 6
max_seq_len = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

with open('input.txt', 'r') as f:
    text = f.read()
print('training data loaded')

chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)
model.load_state_dict(torch.load('nanogpt_model.pt', map_location=device))
model.eval()


@torch.no_grad()
def gen_step(model, token_ids, k_caches, v_caches, temperature, pos_offset, top_k):
    logits, k_caches, v_caches = model(token_ids, k_cache=k_caches, v_cache=v_caches,
                                       infer=True, pos_offset=pos_offset)
    next_logits = logits[0, -1, :] / temperature
    if top_k > 0:
        topk_vals = torch.sort(next_logits)[-top_k:]
        threshold = topk_vals[0]
        next_logits = torch.where(next_logits < threshold, torch.tensor(float('-inf'), device=device), next_logits)
    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, k_caches, v_caches


def generate(model, token_ids, n_new_tokens, temperature=1.0, top_k=25):
    token_ids = token_ids.to(device)
    next_token, k_caches, v_caches = gen_step(model, token_ids, None, None, temperature, 0, top_k)
    print(id_to_char[next_token.item()], end='', flush=True)
    pos = token_ids.shape[1]
    token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)

    for _ in range(n_new_tokens):
        next_token, k_caches, v_caches = gen_step(model, next_token[None, :],
                                                   k_caches, v_caches, temperature, pos, top_k)
        print(id_to_char[next_token.item()], end='', flush=True)
        token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)
        if k_caches[0].shape[2] >= max_seq_len:
            k_caches = [k[:, :, -max_seq_len:, :] for k in k_caches]
            v_caches = [v[:, :, -max_seq_len:, :] for v in v_caches]
        pos += 1
    print('')
    return token_ids


context = torch.tensor([[char_to_id[c] for c in 'JULIET']])
output_ids = generate(model, context, 500, temperature=1.0, top_k=25)