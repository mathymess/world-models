import csv
from typing import Callable
import random

import nltk
from nltk.corpus import wordnet as wn

import torch
from torch.utils.data import Dataset


Synset = nltk.corpus.reader.wordnet.Synset
HyperHypoPair = tuple[Synset, Synset]


def get_all_hyper_hypo_pairs() -> list[HyperHypoPair]:
    hyper_hypo_pairs = []
    for hyper in wn.all_eng_synsets(wn.NOUN):
        for hypo in hyper.hyponyms():
            pair = (hyper, hypo)
            hyper_hypo_pairs.append(pair)
    return hyper_hypo_pairs


def are_related(s: Synset, t: Synset) -> bool:
    lch = s.lowest_common_hypernyms(t)
    return s in lch or t in lch


def generate_unrelated_pairs(how_many_pairs: int = 50000, rng_seed: int = 42) -> list[HyperHypoPair]:
    unrelated_pairs = []
    all_synsets = list(wn.all_eng_synsets(wn.NOUN))
    rng = random.Random(rng_seed)
    while len(unrelated_pairs) < how_many_pairs:
        s, t = rng.sample(all_synsets, k=2)
        if not are_related(s, t):
            unrelated_pairs.append((s, t))
    return unrelated_pairs


def synset_to_str(s: Synset) -> str:
    lemma_name = s.lemmas()[0].name()
    return str(lemma_name).replace("_", " ")


def get_mixed_prompt_dataset_impl(
    true_pairs: list[HyperHypoPair],
    bogus_pairs: list[HyperHypoPair],
    to_string_mapper: Callable[[HyperHypoPair], str]
) -> list[str]:
    return ([(to_string_mapper(p), True) for p in true_pairs]
          + [(to_string_mapper(p), False) for p in bogus_pairs])


def get_mixed_prompt_dataset(
    size: int = 50000,
    prompt_type: str = "is_a_kind_of",
    rng_seed: int = 42
) -> list[tuple[str, bool]]:
    if size < 2:
        raise ValueError(f"too small size: {size}")
    if prompt_type not in ["is_a_kind_of"]:
        raise ValueError(f"unknown prompt type: {prompt_type}")

    all_true_pairs = get_all_hyper_hypo_pairs()
    true_pairs = random.Random(rng_seed).sample(all_true_pairs, k=size // 2)
    bogus_pairs = generate_unrelated_pairs(size - size // 2, rng_seed=rng_seed)

    def to_string_mapper(pair: HyperHypoPair) -> str:
        hyper, hypo = pair
        # TODO: if-else on other prompt_types
        return f"{synset_to_str(hypo)} is a kind of {synset_to_str(hyper)}"

    return get_mixed_prompt_dataset_impl(
        true_pairs,
        bogus_pairs,
        to_string_mapper
    )


def save_prompt_dataset_as_csv(dataset: list[tuple[str, bool]], filepath: str):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prompt", "label"])
        writer.writerows(dataset)


if __name__ == "__main__":
    nltk.download("wordnet")

    d = get_mixed_prompt_dataset()
    print(random.sample(d, k=20))

    filepath = "CHECK_THIS_OUT.csv"
    print(filepath)
    save_prompt_dataset_as_csv(d, filepath)
