import torch
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet as wn


Synset = nltk.corpus.reader.wordnet.Synset


def synset_to_str(s: Synset) -> str:
    lemma_name = s.lemmas()[0].name()
    return str(lemma).replace("_", " ")


class HyperHypoPairsDataset(Dataset):
    def __init__(self):
        nltk.download("wordnet")
        self.hyper_hypo_pairs = []

        for hyper in wn.all_eng_synsets(wn.NOUN):
            for hypo in hyper.hyponyms():
                pair = (hyper, hypo)
                self.hyper_hypo_pairs.append(pair)

    def __getitem__(self, i) -> tuple[Synset, Synset]:
        return self.hyper_hypo_pairs[i]

    def __len__(self) -> int:
        return len(self.hyper_hypo_pairs)


class XIsKindOfYWrapperDataset(Dataset):
    def __init__(self, underlying_dataset: Dataset = HyperHypoPairsDataset()):
        self.underlying_dataset = underlying_dataset

    def __getitem__(self, i) -> str:
        hyper, hypo = self.underlying_dataset[i]
        return f"{synset_to_str(hypo)} is a kind of {synset_to_str(hyper)}"

    def __len__(self) -> int:
        return len(self.underlying_dataset)
