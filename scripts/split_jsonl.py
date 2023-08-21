import argparse
import gzip
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output file")
    args = parser.parse_args()

    total = 0
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            total += 1

    per_file = math.ceil(total / len(args.outputs))

    with gzip.open(args.input, "rt") as ifd:
        for fname in args.outputs:
            with gzip.open(fname, "wt") as ofd:
                for i, line in enumerate(ifd):
                    ofd.write(line)                    
                    if i >= per_file:
                        break
