# Blacksmith

my own AI (Artificial Idiot)...

## training

+ get the things in `requirements.txt`
+ pre-tokenize two datasets:
  + https://huggingface.co/datasets/openbmb/UltraChat/tree/main
  + https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus (the cosmopedia v2)
  + (use `prepare_ultra_chat.py` and `prepare_cosmopediav2.py`)
  + if you don't want to pre-tokenize yourself you can use the files i did:
    + https://huggingface.co/datasets/bctnry/UltraChat
	+ https://huggingface.co/datasets/bctnry/CosmopediaV2
+ change parameters in `hyperparameters.py` to your heart's content
+ run `main.py` using `torchrun`
+ wait (for weeks)



