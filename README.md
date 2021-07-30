# `ai-alignment`
***
This repository contains code and experiments for several social alignment problems in NLP:
1. <a href="https://github.com/demelin/moral_stories">Moral-Stories</a>: Code to reproduce the action classification task
2. <a href="https://github.com/hendrycks/ethics">ETHICS</a>
3. **TODO** MACS

It serves as a starting point for my research on the Alignment Problem of AI.

# Installation
## Windows
1. Clone directory and `cd` into it
2. Create a python environment called `env`:
```shell
python -m venv env
```
3. Run `run_setup.bat`
4. Install `PyTorch`, `Spacy` and `cupy` according to your GPU/CPU needs. E.g. for CUDA 11.2: 
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install spacy[cuda112]
```
4. Start jupyter with `start_jupyter.bat`

## Linux
This remains a TODO until it is needed in the future...

## Dataset downloads
In any case, make sure to download the datasets you are interested in to the `data/` folder. No need to unpack anything!
1. <a href="https://tinyurl.com/y99sg2uq">Download Moral-Stories</a>
2. <a href="https://people.eecs.berkeley.edu/~hendrycks/ethics.tar">Download ETHICS</a>
3. <a href="https://tinyurl.com/y99sg2uq">MACS still TODO</a>