import torch
import tiktoken
import pyarrow.dataset as pds

# converts .parquet datasets into something DatasetManager needs.

print('creating tokenizer...')
Tokenizer = tiktoken.get_encoding('gpt2')
print('opening dataset...')
ds = pds.dataset('bookcorpusopen/data', format='parquet')
result = open('tokens.bin', 'wb')
count = 0
print('loading dataset...')
table = ds.to_table()
for t in table.column('text'):
    print(f'preparing {count+1}/{len(table)}')
    token_list = Tokenizer.encode(t.as_py())
    token_list.append(Tokenizer.eot_token)
    for k in token_list:
        byte1 = k%256
        byte2 = (k//256)%256
        result.write(bytes([byte1, byte2]))
    count += 1

