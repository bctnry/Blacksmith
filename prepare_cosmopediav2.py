import os
import json
import pathlib
import pyarrow.dataset as pds
from transformers import GPT2Tokenizer

# NOTE(bctnry,2026.8.14):
# i've decided to pre-tokenize all of cosmopedia-v2. i do this because (1) it
# saves time; (2) it's easier to memmap; (3) it's hard to slice exactly `max_seq_len`
# tokens out from the dataset (because one token isn't necessary one byte/character).
# this script only extracts the `text` feld. edit the code if you need more.

Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
Tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})

cosmopedia_v2_path = pathlib.Path('/home/bctnry/workspace/datasets/CosmopediaV2/')

ds = pds.dataset(cosmopedia_v2_path, format='parquet')
total = ds.count_rows()
print(f'total: {total}')
batches = ds.to_batches(columns=['text'])

file_count = 0
EOT = Tokenizer.encode('<|endoftext|>')[0]
current_file = open(f'cosmopediav2_{file_count:08}.bin', 'wb')
batch_count = 0
count = 0
for bb in batches:
    batch_count += 1
    print(f'batch {batch_count}')
    b = list(bb)[0]
    for c in b:
        encoded = Tokenizer.encode(str(c))
        encoded.append(EOT)
        for tok in encoded:
            byte1 = tok%256
            byte2 = (tok//256)%256
            current_file.write(bytes([byte1, byte2]))
        count += 1
        if count % 100 == 0:
            print('.', end='', flush=True)
        if count % 1000 == 0:
            print(f'{count / total * 100:.4f}', flush=True)
        if count % 5000000 == 0:
            current_file.close()
            file_count += 1
            current_file = open(f'cosmopediav2_{file_count:08}.bin', 'wb')

current_file.close()


