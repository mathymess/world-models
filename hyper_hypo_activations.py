from hyper_hypo_data import get_mixed_prompt_dataset, save_prompt_dataset_as_csv

from transformers import LlamaForCausalLM, AutoTokenizer

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import einops

import os
import argparse

def tokenize_prompts(prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    encode_res = tokenizer.batch_encode_plus(
        prompts,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        return_attention_mask=True
    )

    token_ids = encode_res['input_ids']
    attention_mask = encode_res['attention_mask']

    return token_ids, attention_mask


def load_model(device="cuda"):
    model = LlamaForCausalLM.from_pretrained(
        f"meta-llama/Llama-2-13b-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.eval()
    model.to()
    return model


@torch.no_grad()
def get_layer_activations_hf(bs: int, tokenized_prompts: torch.Tensor,
                             attention_mask: torch.Tensor, device=None):
    model = load_model()

    layers = list(range(model.config.num_hidden_layers))
    if device is None:
        device = model.device

    n_seq, ctx_len = tokenized_prompts.shape
    activation_rows = n_seq

    layer_activations = {
        l: torch.zeros(activation_rows, model.config.hidden_size, dtype=torch.float16)
        for l in layers
    }

    offset = 0
    dataloader = DataLoader(tokenized_prompts, batch_size=bs, shuffle=False)

    for step, batch in enumerate(tqdm(dataloader, disable=False)):
        # clip batch to remove excess padding
        batch_attention_mask = attention_mask[step*bs:(step+1)*bs]
        last_valid_ix = torch.argmax(
            (batch_attention_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        batch = batch[:, :last_valid_ix].to(device)
        batch_attention_mask = batch_attention_mask[:, :last_valid_ix]

        out = model(batch, output_hidden_states=True,
                    output_attentions=False, return_dict=True, use_cache=False)

        # do not save post embedding layer activations
        for lix, activation in enumerate(out.hidden_states[1:]):
            if lix not in layer_activations:
                continue
            activation = activation.cpu().to(torch.float16)
            processed_activations = process_activation_batch(activation, step, batch_attention_mask)

            save_rows = processed_activations.shape[0]
            layer_activations[lix][offset:offset +
                                   save_rows] = processed_activations

        offset += batch.shape[0]

    return layer_activations


def process_activation_batch(batch_activations, step, batch_mask=None):
    last_ix = batch_activations.shape[1] - 1
    batch_mask = batch_mask.to(int)
    last_entity_token = last_ix - torch.argmax(batch_mask.flip(dims=[1]), dim=1)
    d_act = batch_activations.shape[2]
    expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
    processed_activations = batch_activations[
        torch.arange(cur_batch_size).unsqueeze(-1),
        expanded_mask,
        torch.arange(d_act)
    ]
    assert processed_activations.shape == (cur_batch_size, d_act)
    return processed_activations


def quantize(activation_tensor: torch.Tensor) -> torch.Tensor:
    min_vals = activation_tensor.min(dim=0)[0]
    max_vals = activation_tensor.max(dim=0)[0]

    output_precision = 8
    num_quant_levels = 2 ** output_precision

    scale = (max_vals - min_vals) / (num_quant_levels - 1)
    zero_point = torch.round(-min_vals / scale)
    return torch.quantize_per_channel(activation_tensor, scale, zero_point, 1,  torch.quint8)


def write_activations_to_disk(layer_activations: dict[int, torch.Tensor], save_dir: str):
    for layer_ix, activations in layer_activations.items():
        save_path = os.path.join(save_dir, f"{layer_ix}.pt")
        activations = quantize(activations.to(torch.float32))
        torch.save(activations, save_path)


def main(batch_size: int, save_dir: str):
    prompts, _labels = get_mixed_prompt_dataset(size=50000, prompt_type="is_a_kind_of", rng_seed=42)
    tokenized_prompts, attention_mask = tokenize_prompts(prompts)

    layer_activations = get_layer_activations_hf(batch_size, tokenized_prompts, attention_mask)
    write_activations_to_disk(layer_activations, save_dir)


if __name__ == "__main__":
    main(batch_size=48, save_dir="hyper_hypo_activations0/")
