#!/bin/bash
#python -u save_activations.py --model Llama-2-13b-hf --prompt_name empty --batch_size 48 --entity_type world_place
python -u save_activations.py --model Llama-2-13b-hf --prompt_name empty --batch_size 48 --entity_type us_place
python -u save_activations.py --model Llama-2-13b-hf --prompt_name empty --batch_size 48 --entity_type historical_figure
python -u save_activations.py --model Llama-2-13b-hf --prompt_name empty --batch_size 48 --entity_type headline
python -u save_activations.py --model Llama-2-13b-hf --prompt_name empty --batch_size 48 --entity_type nyc_place