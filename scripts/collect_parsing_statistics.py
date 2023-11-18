import argparse
import gzip
import logging
import json
import numpy
import igraph

logger = logging.getLogger("collect_parsing_statistics")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            logger.info("Processing document #%d", i + 1)
            j = json.loads(line)
            year = j["inferred_year"]
            author = j["author"]
            title = j["title"]
            diams = []
            dep_stddevs = []
            wlens = []
            slens = []
            for sent in j["content"]:
                adj = numpy.zeros((len(sent), len(sent)))
                for i, w in enumerate(sent):
                    if w["governor"] != -1:
                        adj[w["index_token"], w["governor"]] = 1
                    wlens.append(len(w["string"]))
                g = igraph.Graph.Adjacency(adj)
                diam = g.diameter()
                diams.append(g.diameter())
                slens.append(len(sent))
                dep_stddevs.append(adj.sum(0).std())
            ofd.write(
                json.dumps(
                    {
                        "author" : author,
                        "title" : title,
                        "year" : year,
                        "diameters" : diams,
                        "sentence_lengths" : slens,
                        "word_lengths" : wlens,
                        "dependent_stddevs" : dep_stddevs
                    }
                ) + "\n"
            )
