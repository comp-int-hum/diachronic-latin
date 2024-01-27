import sys
import logging
import gzip
import math
import json
import argparse
import numpy as np
import numpy
from detm import DETM
import torch
import pandas


logger = logging.getLogger("generate_topics_summary_table")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument('--top_words', type=int, default=10, help='Number of top words')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device("cpu"))

    token2id = {v : k for k, v in model.id2token.items()}
    id2token = model.id2token
    model.to("cpu")

    topics = []
    with torch.no_grad():
        model.mu_q_alpha = model.mu_q_alpha.cpu()
        alpha = model.mu_q_alpha.cpu().numpy()
        model.rho = model.rho.cpu()
        beta = model.get_beta(model.mu_q_alpha).cpu().numpy()
        beta = beta[:, 3, :]
        sh = beta.shape
        #print(beta.sum(1).sum(1))
        #print(beta.max())
        #beta = beta.sum(1) / sh[1]
        #beta = beta.sum(1)
        for row in beta:
            topic = []
            for i in list(reversed(numpy.argsort(row)))[0:args.top_words]:
                topic.append("{}:{:.02f}".format(id2token[i], row[i]))                
            topics.append(topic)
        
    pd = pandas.DataFrame(topics)
    with open(args.output, "wt") as ofd:
        ofd.write(pd.to_latex(index=False, index_names=False))


