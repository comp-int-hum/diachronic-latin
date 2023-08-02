import argparse
import json
import gzip
import sys
import csv
import re

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--min_year", dest="min_year", type=int, default=-500, help="Minimum year")    
    parser.add_argument("--max_year", dest="max_year", type=int, default=500, help="Maximum year")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    year_rx = r"((?P<century>[a-z0-9]+)(st|nd|rd|th)\s+cent|(?P<year>(\d+)))"
    
    counts = {}
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        c_ifd = csv.DictReader(
            ifd,
            delimiter="\t"
        )
        for row in c_ifd:
            dates = []
            if re.match(r".*\D\d{4}\D.*", row["author"]):
                continue
            for m in re.finditer(year_rx, row["author"], flags=re.I):
                d = m.groupdict()
                if d["century"]:
                    dates.append(int(d["century"]) * 100 - 50)
                else:
                    dates.append(int(d["year"]))
            
            if len(dates) > 0:
                bc = True and re.match(r".*(^|\s)b\.?c\.?e?($|\s).*", row["author"], flags=re.I)
                date = (-1 if bc else 1) * dates[-1]
                if date <= args.max_year and date >= args.min_year:
                    counts[row["author"]] = counts.get(row["author"], {"count" : 0, "year" : date})
                    counts[row["author"]]["count"] += 1
                    row["inferred_year"] = date
                    ofd.write(json.dumps(row) + "\n")
