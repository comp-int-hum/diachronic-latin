import argparse
import math
import gzip
import json
import sys
import pickle
import pandas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--plots", dest="plots", help="Output file")
    parser.add_argument("--tables", dest="tables", help="Output file")
    args = parser.parse_args()

    syntax_labels = {
        "diameters" : "Normalized parse tree diameter",
        "sentence_lengths" : "Sentence length",
        "word_lengths" : "Word length",
        "dependent_stddevs" : "Standard deviation of dependent count",
    }
    
    with gzip.open(args.input, "rb") as ifd:
        matrices = pickle.loads(ifd.read())

    start = matrices["start"]
    window_size = matrices["window_size"]
    id2author = matrices["id2author"]
    id2word = matrices["id2word"]
    wwt = matrices["wwt"]
    awt = matrices["awt"]
    syntax = matrices["syntax"]

    fig = Figure(figsize=(24, 24))

    ax = fig.add_subplot(2, 1, 1)
    #ax.plot()
    
    with gzip.open(args.plots, "wb") as ofd:
        fig.savefig(ofd, bbox_inches="tight")
