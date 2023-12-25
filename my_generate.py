import torch
from torch.nn import functional as F
from make_prompt_datasets import load_tokenizer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

tokenizer = load_tokenizer("Llama-2-7b-hf")
next_token_logits = out.logits[:, -1, :]
# filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
generated = torch.cat([inp, next_token], dim=-1)
resulting_string = tokenizer.decode(generated.tolist()[0])
