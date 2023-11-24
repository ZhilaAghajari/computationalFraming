import numpy as np
import json
import argparse
import os
import random
random.seed(2023)

import gensim.corpora as corpora
from tqdm import tqdm
from model import ThetaRoleModel

def parse_args():
    parser = argparse.ArgumentParser()

    # n topics and n latent theta roles
    parser.add_argument('--K', nargs='?', type=int, default=5)
    parser.add_argument('--T', nargs='?', type=int, default=2)

    # dirichlet initialization hyper parameters (static)
    parser.add_argument('--alpha', nargs='?', type=float, default=0.1) 
    parser.add_argument('--eta', nargs='?', type=float, default=0.1) 
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1) 
    parser.add_argument('--lam', nargs='?', type=float, default=0.1)
    parser.add_argument('--n_iters', nargs='?', type=int, default=20)
    parser.add_argument('--corpus_path', nargs='?', type=str, default="data/processed/corpus.json")

    args = parser.parse_args()

    return args.K, args.T, args.alpha, args.eta, args.gamma, args.lam, args.n_iters, args.corpus_path

def main():
    K, T, alpha, eta, gamma, lam, n_iters, corpus_path = parse_args()

    # [TODO]: change to BSON instead of JSON for faster io and smaller storage
    with open(corpus_path) as json_file:
        o = json.load(json_file)

    doc_objects = o['documents']
    docs = [ doc_objects[str(doc_id)]['words'] for doc_id in doc_objects ]
    doc_relns = [ doc_objects[str(doc_id)]['relns'] for doc_id in doc_objects ] #what is this 
    vocab = o['vocab']
    vocab_relns = o['vocab_relns']

    # document preprocessing helpers
    id2word = corpora.Dictionary(docs)
    reln2id = {reln:i for i, reln in enumerate(vocab_relns)}
    corpus = list(map(lambda x: id2word.doc2idx(x), docs))

    # initialize scalars from plate diagram
    D, V, R = len(docs), len(vocab), len(vocab_relns) # n documents, n words, n relns: https://universaldependencies.org/u/dep/

    # initialize theta role model
    theta_model = ThetaRoleModel(corpus, doc_relns, vocab_relns, id2word, reln2id, n_iters, K, T, D, V, R, alpha, eta, gamma, lam)
    theta_model.initialize_variables()
    theta_model.fit()

    # compute matrices
    theta_model.compute_matrices()

    # print topics, theta roles, and top topics/theta roles for each document
    theta_model.print_all()


if __name__ == "__main__":
    main()
