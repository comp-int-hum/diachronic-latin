import json
import gzip
import argparse
from cltk.lexicon.lat import LatinLewisLexicon

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    lex = LatinLewisLexicon()
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            sents = []
            for sent in j["content"]:
                sents.append(
                    [t["lemma"] for t in sent if t["lemma"] in lex.entries]
                )
            j["content"] = sents
            ofd.write(json.dumps(j) + "\n")
            print(i)
