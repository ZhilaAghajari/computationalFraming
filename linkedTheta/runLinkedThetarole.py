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
    parser.add_argument('--K', nargs='?', type=int, default=10)
    parser.add_argument('--T', nargs='?', type=int, default=5)

    # dirichlet initialization hyper parameters (static)
    parser.add_argument('--alpha', nargs='?', type=float, default=0.1) 
    parser.add_argument('--eta', nargs='?', type=float, default=0.1) 
    parser.add_argument('--etaprime', nargs='?', type=float, default=0.1)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1)
    parser.add_argument('--omega', nargs='?', type=float, default=0.1)
    parser.add_argument('--lam', nargs='?', type=float, default=0.1)
    parser.add_argument('--n_iters', nargs='?', type=int, default=1000)
    parser.add_argument('--corpus_path', nargs='?', type=str, default="threeQuarter_DoH.json")
    parser.add_argument('--corpusName', nargs='?', type=str, default="threeQuarter_DoH")
#     parser.add_argument('--corpus_path', nargs='?', type=str)
#     parser.add_argument('--corpusName', nargs='?', type=str)

    
#     parser.add_argument('--corpus_path', nargs='?', type=str, default="sample_15_prepCollapsed.json")
#     parser.add_argument('--corpusName', nargs='?', type=str, default="sample_15_prepCollapsed")
    args = parser.parse_args()

    return args.K, args.T, args.alpha, args.eta, args.etaprime, args.gamma, args.lam, args.omega , args.n_iters, args.corpus_path, args.corpusName





def main():
    K, T, alpha, eta, etaprime, gamma, lam, omega, n_iters, corpus_path, corpusName= parse_args()

    # [TODO]: change to BSON instead of JSON for faster io and smaller storage
    with open(corpus_path) as json_file:
        o = json.load(json_file)

    print('corpus file loaded...')
    doc_objects = o['documents'] #words, reln, originaltext
    docs = [ doc_objects[str(doc_id)]['words'] for doc_id in doc_objects ]
    text = [ doc_objects[str(doc_id)]['originaltext'] for doc_id in doc_objects ]
    doc_relns = [ doc_objects[str(doc_id)]['relns'] for doc_id in doc_objects ]
    doc_arg2 = [ doc_objects[str(doc_id)]['arg2'] for doc_id in doc_objects ]
    vocab = o['vocab']
    vocab_relns = o['vocab_relns']
    vocab_arg = o['vocab_arg']

    
    
    # document preprocessing helpers
    id2word = corpora.Dictionary(docs)
    reln2id = {reln:i for i, reln in enumerate(vocab_relns)}
    arg2id = {arg:i for i, arg in enumerate(vocab_arg)}
    corpus = list(map(lambda x: id2word.doc2idx(x), docs))
    originaltext = {doc_id: doc_objects[str(doc_id)]['originaltext'] for doc_id in doc_objects}
    idx2docid =  {i:doc_id for i, doc_id in enumerate(doc_objects)}


    # initialize scalars from plate diagram
    D, V, R, A2= len(docs), len(vocab), len(vocab_relns), len(vocab_arg) # n documents, n words, n relns: https://universaldependencies.org/u/dep/
    
    
    # initialize theta role model
    
    theta_model = ThetaRoleModel(corpus, originaltext, doc_relns, doc_arg2, vocab_relns, vocab_arg, id2word, reln2id, arg2id, idx2docid, n_iters, K, T, D, V, R, A2, alpha, eta, etaprime, gamma, lam, omega, corpusName)
    theta_model.initialize_variables()
    theta_model.fit()
    
    
    # compute matrices
    theta_model.compute_matrices()

    # print topics, theta roles, and top topics/theta roles for each document.. how to put them into a file instead of print
    theta_model.print_all()


if __name__ == "__main__":
    main()
