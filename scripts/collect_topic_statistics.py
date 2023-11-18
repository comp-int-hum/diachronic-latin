import argparse
import gzip
import json
import logging

logger = logging.getLogger("collect_topic_statistics")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            if i > 10:
                break
            j = json.loads(line)
