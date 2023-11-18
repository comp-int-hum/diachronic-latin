import json
import gzip
import re
import argparse
from cltk.lexicon.lat import LatinLewisLexicon
import unicodedata

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--prefixes_to_preserve", dest="prefixes_to_preserve", nargs="*", default=[])
    parser.add_argument("--include_greek", dest="include_greek", default=False, action="store_true")
    args = parser.parse_args()

    ew_rx = "^({})".format("|".join(["{}.*".format(w) for w in args.prefixes_to_preserve]))
    
    lex = LatinLewisLexicon()
    cache = {}
    def test_token(t):
        return (
            (t["lemma"] != "que") and
            (
                t["lemma"] in lex.entries or
                #re.match(ew_rx, t["lemma"]) or
                (args.include_greek and all([unicodedata.name(c).startswith("GREEK") for c in t["lemma"]]))
            )
        )                
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            sents = []
            for sent in j["content"]:
                sents.append(
                    [re.sub(r"que$", "", re.sub(r"[\d.,-]", "", t["lemma"])) for t in sent if cache.setdefault(t["lemma"], test_token(t))]
                )
            j["content"] = sents
            ofd.write(json.dumps(j) + "\n")
