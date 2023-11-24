import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from tqdm import tqdm
import json
from datasets import load_dataset
from gensim import utils
from collections import defaultdict

from word import Word
from malt.malt import MaltParser # source code from nltk library

def process_dataset(docs, stop_words, malt_parser_version='maltparser-1.7.2', model_version='engmalt.linear-1.7.mco'):
    # initalize malt parser model
    mp = MaltParser(malt_parser_version, model_version, tagger=nltk.pos_tag)

    # create <doc_idx, tokenized_sent> list of sents
    sents = [
        (i, nltk.word_tokenize(sent))
        for i, doc in enumerate(docs)
        for sent in nltk.sent_tokenize(utils.to_unicode(str(doc).lower())) # convert doc to lowercase, and sentence tokenized.
    ]

    # unzip list of tuples
    doc_idxs, sents = zip(*sents)

    # create parser <generator> and loop through parser to produce dependency tree for each sentence
    parser = mp.parse_sents(sents, verbose=True)

    # define valid word
    valid_word = lambda word: not word in stop_words and word.isalpha() and len(word) > 2

    # initalize dictionary for json output
    docs_dict = {
        'documents': dict((doc_idx, {'words': [], 'relns': []}) for doc_idx in doc_idxs),
        'vocab': [], 
        'vocab_relns': [],
    }
    # initalize vocab variables as sets (no duplicates)
    vocab = set()
    vocab_relns = set()

    i = 0
    # loop through list iterators
    for list_it in parser:
        tree = next(list_it)
        # check if valid tree, if not skip
        try:
            nodes = tree.nodes
        except:
            continue
        
        word_relns_hash = defaultdict(list)
        for word_idx in nodes:
            if word_idx == 0: # skip first
                continue
            
            deps = nodes[word_idx]['deps']

            # check for valid dependency relations
            if deps:
                for reln, idxs in deps.items():
                    for idx in idxs:
                        dep_reln, gov_reln = f"{reln}.dep", f"{reln}.gov"
                        # add relations to vocab
                        vocab_relns.add(dep_reln)
                        vocab_relns.add(gov_reln)

                        # add reln to word in hashmap
                        word_relns_hash[idx].append(dep_reln) # append to dep word
                        word_relns_hash[word_idx].append(gov_reln) # append to current word
        
        # check for valid hashmap
        if word_relns_hash:
            doc_idx = doc_idxs[i]

            # loop through hashmap items and append to dict for future storing
            for word_idx, relns in word_relns_hash.items():
                word = nodes[word_idx]['word']
                relns = [reln for reln in relns if reln != "punct.gov"]
                if valid_word(word):
                    vocab.add(word)
                    docs_dict['documents'][doc_idx]['words'].append(word)
                    docs_dict['documents'][doc_idx]['relns'].append(relns)
        
        i += 1

    docs_dict['vocab'] = list(vocab)
    docs_dict['vocab_relns'] = list(vocab_relns)

    return docs_dict


def main():
    # [UNCOMMENT] uncomment these lines to use own dataset, 2 lines after used for testing
    # data = pd.read_csv("data/raw/text_data.csv")
    # corpus = data['text']
    dailymail = load_dataset('cnn_dailymail', '2.0.0') # https://huggingface.co/datasets/cnn_dailymail/viewer/2.0.0/
    corpus = dailymail['train']['article'][:200]

    # stop word initialization
    with open("data/utils/stopwords.txt") as f:
        more_stop_words = f.read().splitlines()
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(more_stop_words)
    
    # obtain word relation pairs
    docs_dict = process_dataset(corpus, stop_words)
    # convert to json object
    json_object = json.dumps(docs_dict, indent=4)
    #Here, we need to also add the original documents... 

    # store json object
    with open("data/processed/corpus.json", "w") as f:
        f.write(json_object)

if __name__ == "__main__":
    main()