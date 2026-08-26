# Blacksmith

> this location is currently deprecated; for further developments please check [https://github.com/TriusAI/Blacksmith](https://github.com/TriusAI/Blacksmith)

my own AI (Artificial Idiot)...

## training

+ get the things in `requirements.txt`
+ pre-tokenize two datasets:
  + https://huggingface.co/datasets/openbmb/UltraChat/tree/main
  + https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus (the cosmopedia v2)
  + (use `prepare_ultra_chat.py` and `prepare_cosmopediav2.py`)
  + if you don't want to pre-tokenize yourself you can use the files i did:
    + https://huggingface.co/datasets/bctnry/UltraChat
	+ (i'd upload cosmopedia v2 to https://huggingface.co/datasets/bctnry/CosmopediaV2 sometimes but not now)
+ change parameters in `hyperparameters.py` to your heart's content
+ run `main.py` using `torchrun`
+ wait (for weeks)
+ after finishing training the base model, prepare `identity.jsonl`
+ also prepare a part of (or all of) UltraChat in its .jsonl form
+ run `sft.py` using `torchrun`
+ wait (for hours)
+ run `generate.py`

## running

+ get the `.pt` file from [https://huggingface.co/bctnry/Blacksmith/tree/main/Blacksmith-1-20260827](https://huggingface.co/bctnry/Blacksmith/tree/main/Blacksmith-1-20260827)
+ run `generate.py`
+ enjoy this artificial idiot which can barely answer any questions at all

## further plans

+ find better data
+ train bigger models
+ learn quantization
+ try out new architectures (e.g. [simple attention networks](https://arxiv.org/abs/2607.18363)

## notes

+ because we're using huggingface's library it's recommended to first
  download the gpt2 tokenizer separately by running this in python
  (after installing `transformers`):
  
  ``` python
  from transformers import GPT2Tokenizer
  GPT2Tokenizer.from_pretrained('gpt2')
  ```
  
  and *only then* use `HF_HUB_OFFLINE=1`. i'm currently living in a
  place with peculiar network situations so i have to do this kind of
  maneuver, but you might not need to.
+ `main.py` trains with no attention mask; we always get full
  `max_seq_len` sequences via `MixedDatasetManager` (in
  `dataset_mixed.py`), so there's no padding and no need for masking.
  SFT is different: conversations vary in length and can have very
  short sequences, so padding is needed for those cases. without an
  attention mask the model would attend to padding tokens during SFT,
  learning attention patterns that depend on padding being present,
  but `generate.py` feeds unpadded sequences at inference, so those
  patterns would break and the model would produce garbage. `sft.py`
  passes an attention mask so the model ignores padding positions,
  keeping training and inference conditions consistent.
+ the attention mask combination in `model.py` uses the boolean form
  (`True = masked/excluded`) rather than the additive float form.
  this is deliberate: in bfloat16, the additive form overflows
  (`finfo.min + finfo.min = -inf`), which produces NaN in softmax and
  silently destroys training. the boolean form has no such issue.
+ `sft_dataset.py` truncates long conversations from the front, not
  the back. this keeps the final assistant response intact so the
  model learns complete responses rather than mid-sentence fragments.
  conversations that can't fit even a single user+assistant turn
  within `max_seq_len` are dropped entirely.

