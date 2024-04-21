Forked the original implementation of the paper (see below) to train my own classifier probes on a different task.
The task is, given a pair of phrases, e.g. "dog" and "animal", to tell if there is an [`is-a`](https://en.wikipedia.org/wiki/Is-a) relationship between them.
Such pairs were taken from WordNet.
Turns out LLaMa-2 can confidently tell if “cow is kind of animal” or if “cat is not a kind of dog”, i.e. identify true hyponym-hypernym pairs.
See the `hyper_hypo_probing.ipynb` notebook to check out the result.

# World Models in LLMs

Official code repository for the paper "Language Models Represent Space and Time" by Wes Gurnee and Max Tegmark.

This repository contains all experimental infrastructure for the paper. We expect most users to just be interested in the cleaned data CSVs containing entity names and relevant metadata. These can be found in `data/entity_datasets/` (with the tokenized versions for Llama and Pythia models available in the `data/prompt_datasets/` folder for each prompt type).

In the coming weeks we will release a minimal version of the code to run basic probing experiments on our datasets.

## Cite

If you found our code our datasets helpful in your research, please cite our paper
```
@article{gurnee2023language,
  title={Language Models Represent Space and Time},
  author={Gurnee, Wes and Tegmark, Max},
  journal={arXiv preprint arXiv:2310.02207},
  year={2023}
}
```
