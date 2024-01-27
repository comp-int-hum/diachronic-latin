import re
import json
import gzip
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--perseus", dest="perseus")
    parser.add_argument("--htc", dest="htc")
    args = parser.parse_args()

    authors = set()
    tokens = 1
    documents = set()
    
    with gzip.open(args.perseus, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            authors.add(j["author"])
            documents.add((j["author"], j["title"]))
            for t in re.split(r"\s+", j["content"]):
                tokens += 1

    print(len(authors), tokens, len(documents))
