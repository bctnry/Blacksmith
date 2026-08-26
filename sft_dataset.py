import json
import torch
import logging
from torch.utils.data import Dataset
from tokenmaker import Tokenizer

logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

IM_START = Tokenizer.encode('<|im_start|>')[0]
IM_END = Tokenizer.encode('<|im_end|>')[0]
EOT = Tokenizer.eos_token_id  # 50256
NEWLINE = Tokenizer.encode('\n')[0]
PAD_ID = EOT


def encode_turn(role, content):
    prefix = Tokenizer.encode(f'<|im_start|>{role}\n')
    body = Tokenizer.encode(content)
    suffix = [IM_END, NEWLINE]
    return prefix, body, suffix


def encode_conversation(turns):
    token_ids = []
    labels = []
    for role, content in turns:
        prefix, body, suffix = encode_turn(role, content)
        token_ids.extend(prefix)
        token_ids.extend(body)
        token_ids.extend(suffix)
        if role == 'assistant':
            labels.extend([-100] * len(prefix))
            labels.extend(body)
            labels.extend(suffix)
        else:
            labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
    return token_ids, labels


class SFTDataset(Dataset):
    def __init__(self, sources, max_seq_len=1024, drop_truncated=True):
        self.max_seq_len = max_seq_len
        self.samples = []
        for source in sources:
            path = source[0]
            repeat = source[1] if len(source) > 1 else 1
            file_samples = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    messages = obj.get('data') or obj.get('messages')
                    if not messages:
                        continue
                    turns = []
                    for i, m in enumerate(messages):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        turns.append((role, m))

                    encoded_turns = []
                    total_len = 0
                    for role, content in turns:
                        prefix, body, suffix = encode_turn(role, content)
                        tlen = len(prefix) + len(body) + len(suffix)
                        encoded_turns.append((role, prefix, body, suffix, tlen))
                        total_len += tlen

                    if total_len <= max_seq_len:
                        token_ids = []
                        labels = []
                        for role, prefix, body, suffix, _ in encoded_turns:
                            token_ids.extend(prefix)
                            token_ids.extend(body)
                            token_ids.extend(suffix)
                            if role == 'assistant':
                                labels.extend([-100] * len(prefix))
                                labels.extend(body)
                                labels.extend(suffix)
                            else:
                                labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
                    elif drop_truncated:
                        # Keep the last turns that fit; drop from the front
                        # so the final assistant response stays intact.
                        kept = []
                        kept_len = 0
                        for role, prefix, body, suffix, tlen in reversed(encoded_turns):
                            if kept_len + tlen > max_seq_len:
                                break
                            kept.append((role, prefix, body, suffix))
                            kept_len += tlen
                        kept.reverse()
                        # Must start with a user turn for a valid conversation.
                        if not kept or kept[0][0] != 'user':
                            continue
                        token_ids = []
                        labels = []
                        for role, prefix, body, suffix in kept:
                            token_ids.extend(prefix)
                            token_ids.extend(body)
                            token_ids.extend(suffix)
                            if role == 'assistant':
                                labels.extend([-100] * len(prefix))
                                labels.extend(body)
                                labels.extend(suffix)
                            else:
                                labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
                    else:
                        continue

                    shifted = []
                    for i in range(len(token_ids) - 1):
                        shifted.append(labels[i + 1])
                    shifted.append(-100)
                    labels = shifted

                    file_samples.append((token_ids, labels))

            for _ in range(repeat):
                self.samples.extend(file_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, labels = self.samples[idx]
        pad_len = self.max_seq_len - len(token_ids)
        input_ids = token_ids + [PAD_ID] * pad_len
        labels = labels + [-100] * pad_len
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def make_collate_fn(max_seq_len):
    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return input_ids, labels
    return collate