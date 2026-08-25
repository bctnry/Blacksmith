import torch
import torch.nn.functional as F
from model import NanoGPT
from hyperparameters import d_model, n_heads, n_layers, max_seq_len
from tokenmaker import Tokenizer, TokenizerVocabSize

# we use the format `<|im_start|>{role}\n{content}<|im_end|>`, e.g.:
# 
#     User: What color is the sky?
#     Assistant: Blue, when it's sunny and the sun is not blocked
#                by the clouds.
# 
# would be encoded as:
# 
#     <|im_start|>user
#     What color is the sky?<|im_end|>
#     <|im_start|>assistant
#     Blue, when it's sunny and the sun is not blocked by the clouds.<|im_end|>

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

vocab_size = TokenizerVocabSize

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)
chkp = torch.load('blacksmith-1_model.pt', map_location=device)
model.load_state_dict(chkp['model'])
model.eval()

@torch.no_grad()
def gen_step(model, token_ids, k_caches, v_caches, temperature, pos_offset, top_k):
    logits, k_caches, v_caches = model(token_ids, k_cache=k_caches, v_cache=v_caches,
                                       infer=True, pos_offset=pos_offset)
    next_logits = logits[0, -1, :] / temperature
    if top_k > 0:
        topk_vals = torch.topk(next_logits, top_k).values
        threshold = topk_vals[-1]
        next_logits = torch.where(next_logits < threshold, torch.tensor(float('-inf'), device=device), next_logits)
    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, k_caches, v_caches

END_OF_TEXT = Tokenizer.encode('<|endoftext|>')[0]
END_OF_IM = Tokenizer.encode('<|im_end|>')[0]

def generate(model, token_ids, n_new_tokens, temperature=1.0, top_k=25):
    token_ids = token_ids.to(device)
    next_token, k_caches, v_caches = gen_step(model, token_ids, None, None, temperature, 0, top_k)
    print(Tokenizer.decode(list(token_ids[0])), end='', flush=True)
    pos = token_ids.shape[1]
    token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)

    for _ in range(n_new_tokens):
        next_token, k_caches, v_caches = gen_step(model, next_token[None, :],
                                                   k_caches, v_caches, temperature, pos, top_k)
        if next_token.item() == END_OF_TEXT or next_token.item() == END_OF_IM: break
        print(Tokenizer.decode([next_token.item()]), end='', flush=True)
        token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)
        if k_caches[0].shape[2] >= max_seq_len:
            k_caches = [k[:, :, -max_seq_len:, :] for k in k_caches]
            v_caches = [v[:, :, -max_seq_len:, :] for v in v_caches]
        pos += 1
    print('')
    return token_ids

context = torch.tensor([Tokenizer.encode('<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n')])
output_ids = generate(model, context, 500, temperature=1.0, top_k=25)

