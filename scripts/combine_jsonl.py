import argparse
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", dest="inputs", nargs="+", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        for fname in args.inputs:
            with gzip.open(fname, "rt") as ifd:
                for line in ifd:
                    ofd.write(line)
