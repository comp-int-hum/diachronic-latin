import argparse
import json
import gzip
import sys
import csv
import re

csv.field_size_limit(sys.maxsize)
fields = ["htid", "access", "rights", "ht_bib_key", "description", "source", "source_bib_num", "oclc_num", "isbn", "issn", "lccn", "title", "imprint", "rights_reason_code", "rights_timestamp", "us_gov_doc_flag", "rights_date_used", "pub_place", "lang", "bib_fmt", "collection_code", "content_provider_code", "responsible_entity_code", "digitization_agent_code", "access_profile_code", "author"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hathitrust_index", dest="hathitrust_index", help="")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()
    
    with gzip.open(args.hathitrust_index, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        c_ifd = csv.DictReader(
            ifd,
            fieldnames=fields,
            delimiter="\t"
        )
        c_ofd = csv.DictWriter(
            ofd,
            fieldnames=fields,
            delimiter="\t"
        )
        c_ofd.writeheader()
        for row in c_ifd:
            if row["lang"] == "lat":
                c_ofd.writerow(row)
