# Theta Role Model
## Description
A new approach for leveraging hierarchical topic modeling techniques to analyze and compare dominant frames found during major current events. The latent theta role model is a computational approach to framing analysis that develops latent variables in the form of distribution over words and distributions over grammatical relations to help understand the link between words and grammatical relations.

## Before Running
Preprocess corpus:
```bash
python src/theta/data.py
```
## Running
```bash
python src/theta/run.py [-h] [--K [K]] [--T [T]] [--alpha [ALPHA]] [--eta [ETA]] [--gamma [GAMMA]] [--lam [LAM]] [--n_iters [N_ITERS]] [--corpus_path [CORPUS_PATH]]
```
optional arguments:
```
-h, --help                    show this help message and exit
--K [K]                       number of topics
--T [T]                       number of theta roles
--alpha [ALPHA]               hyperparameter for dirichlet distribution
--eta [ETA]                   hyperparameter for dirichlet distribution
--gamma [GAMMA]               hyperparameter for dirichlet distribution
--lam [LAM]                   hyperparameter for dirichlet distribution
--n_iters [N_ITERS]           number of iterations for gibbs sampling
--corpus_path [CORPUS_PATH]   path to preprocessed corpus 
```

<!-- 
python /runLinedThetarole.py [-h] [--K [K]] [--T [T]] [--alpha [ALPHA]] [--eta [ETA]] [--etaprime [ETAPRIME]] [--gamma [GAMMA]] [--lam [LAM]] [--omega [OMEGA]] [--n_iters [N_ITERS]] [--corpus_path [CORPUS_PATH]], [--corpusName [CORPUS_NAME]] -->

## python src/theta/run.py --k 22 --T 2424 --alpha 242