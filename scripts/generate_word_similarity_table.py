import sys
import logging
import gzip
import math
import json
import argparse
import numpy as np
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from detm import DETM
import pandas
import torch
from gensim.models import Word2Vec
import numpy as np
import numpy



logger = logging.getLogger("generate_word_similarity_table")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", dest="model", help="Model file")
    #parser.add_argument("--embeddings", dest="embeddings", help="W2V embeddings file")
    parser.add_argument("--output", dest="output", help="File to save table")
    parser.add_argument("--top_neighbors", dest="top_neighbors", default=10, type=int, help="How many neighbors to return")
    parser.add_argument('--target_words', default=[], nargs="*", help='Words to consider')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device("cpu"))

    token2id = {v : k for k, v in model.id2token.items()}
    id2token = model.id2token
    #print(len(id2token), len(token2id))
    model.to("cpu")
    try:
        embs = list(model.rho.parameters())[0].detach().numpy()
    except:
        embs = model.rho.cpu().detach().numpy()

    #w2v = Word2Vec.load(args.embeddings)
        
    sims = cosine_similarity(embs)
    #w2v_sims = cosine_similarity(w2v.wv)

    neighbors = []
    # w2v_neighbors, neighbors = [], []
    # words = []
    for w in args.target_words:
        i = token2id[w]
        row = [w]
        for j in list(reversed(numpy.argsort(sims[i]).tolist()))[1:1 + args.top_neighbors]:
            ow = id2token[j]
            op = sims[i][j]
            row.append("{}:{:.02f}".format(ow, op))
        neighbors.append(row)

    #     w2v_row = [w]
    #     for ow, op in w2v.wv.most_similar(w, topn=args.top_neighbors):
    #         w2v_row.append("{}:{:.02f}".format(ow, op))
    #     w2v_neighbors.append(w2v_row)
        
    pd = pandas.DataFrame(neighbors)
    #w2v_pd = pandas.DataFrame(w2v_neighbors)
    with open(args.output, "wt") as ofd:
        ofd.write(pd.to_latex(index_names=False, index=False, header=False))
        #ofd.write("\n\n")
        #ofd.write(w2v_pd.to_latex(index_names=False, index=False, header=False))
