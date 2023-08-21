import argparse
import tarfile
import json
import gzip
import re
import numpy

def split_windows(seq, num_topics):
    per = len(seq) // num_topics
    return [seq[i * per : (i + 1) * per] for i in range(num_topics)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Input file")
    parser.add_argument("--vocab", dest="vocab", help="Input file")
    parser.add_argument("--metadata", dest="metadata", help="Input file")
    parser.add_argument("--text", dest="text", help="Input file")
    parser.add_argument("--time", dest="time", help="Input file")
    parser.add_argument("--docs", dest="docs", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.vocab, "rt") as ifd:
        vocab = json.loads(ifd.read())

    topics = []
    with tarfile.open(args.model, "r") as tfd:
        for mem in tfd.getmembers():
            m = re.match(r".*topic-(\d+)-var-e-log-prob.dat", mem.name)
            if m:
                with tfd.extractfile(mem) as ifd:
                    topics.append([float(l) for l in ifd])
            elif mem.name.endswith("/info.dat"):
                with tfd.extractfile(mem) as ifd:
                    for line in ifd:
                        line = line.decode("utf-8")
                        toks = line.strip().split()
                        if toks[0] == "NUM_TOPICS":
                            num_topics = int(toks[1])
                        if toks[0] == "NUM_TERMS":
                            num_words = int(toks[1])
                        if toks[0] == "SEQ_LENGTH":
                            num_windows = int(toks[1])
    topics = numpy.array([split_windows(t, num_windows) for t in topics])
    topics = numpy.exp(topics)
    summed_topics = topics.sum(1)
    rvocab = {v : k for k, v in vocab.items()}
    top_words = [[rvocab[i] for i in t] for t in numpy.flip(summed_topics.argsort(1), 1)[:, :10].tolist()]

    with gzip.open(args.metadata, "rt") as ifd:
        for line in ifd:
            pass

    prob = topics[:, :, vocab["fabula"]]
    denom = prob.sum(0)
    norm = prob / denom.T

    with open(args.output, "wt") as ofd:
        for i in range(len(top_words)):
            ofd.write("{}: {}\n{}\n\n".format(i + 1, top_words[i], numpy.array2string(norm[i], precision=3)))
