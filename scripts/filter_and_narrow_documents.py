import argparse
import gzip
import re
import json
import math


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lid", dest="lid", help="Input file")
    parser.add_argument("--content", dest="content", help="Input file")    
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--tokens_per_chunk", dest="tokens_per_chunk", type=int, default=200, help="")
    args = parser.parse_args()

    keep = {}
    with gzip.open(args.lid, "rt") as ifd:
        for row in ifd:
            j = json.loads(row)
            key = (j["author"], j["title"])
            if len(j["language_id"]) > 0:
                langs = [x["language_probabilities"][0][0] for x in j["language_id"]]
                lat_prop = len([x for x in langs if x == "la"]) / len(langs)
                if lat_prop > 0.7:
                    cur = keep.get(key, [0.0])[0]
                    if lat_prop > cur:
                        langs = list(reversed(j["language_id"]))
                        while len(langs) > 0 and (langs[0]["language_probabilities"][0][0] != "la" or langs[0]["digit_proportion"] > 0.05 or langs[0]["punctuation_proportion"] > 0.05):
                            langs = langs[1:]
                        j["language_id"] = list(reversed(langs))
                        keep[key] = [lat_prop, j]

    keep = {j["htid"] : j for _, (_, j) in keep.items()}

    
    with gzip.open(args.content, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            if j["htid"] not in keep:
                continue
            toks = re.split(r"\s+", j["content"])
            span_count = math.ceil(len(toks) / args.tokens_per_chunk)
            chunks = [" ".join(toks[i * args.tokens_per_chunk : (i + 1) * args.tokens_per_chunk]) for i in range(span_count)]
            chunks_to_keep = []
            preamble = True
            for chunk, scores in zip(chunks, keep[j["htid"]]["language_id"]):
                if preamble == False or (scores["language_probabilities"][0][0] == "la" and scores["punctuation_proportion"] < 0.05 and scores["digit_proportion"] < 0.05):
                    preamble = False
                    chunks_to_keep.append(chunk)
            j["content"] = " ".join(chunks_to_keep)
            ofd.write(json.dumps(j) + "\n")
