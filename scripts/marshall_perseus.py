import argparse
import re
import json
import gzip
import zipfile
from glob import iglob
import os.path
import lxml.etree as et

#from wikidata.client import Client
#import xml.etree.ElementTree as et

#tree = ET.parse('cic.fam_lat.xml', parser=parser)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--perseus_root", dest="perseus_root", help="Root of Perseus data repos")
    parser.add_argument("--date_ranges", dest="date_ranges")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    authors = {}
    titles = {}
    author_titles = {}
    with open(args.date_ranges, "rt") as ifd:
        for line in ifd:
            if not line.startswith("#"):
                j = json.loads(line)
                date = int(j["start"] + ((j["end"] - j["start"]) / 2))
                if "authors" in j:
                    authors[j["authors"][0]] = date
                elif "author" in j:
                    author_titles[(j["author"], j["title"])] = date
                else:
                    titles[j["title"]] = date

    parser = et.XMLParser(recover=True)

    with gzip.open(args.output, "wt") as ofd:
        for fname in iglob("{}/**/*.xml".format(args.perseus_root), recursive=True):
            with open(fname, "rt") as ifd:
                xml = et.parse(ifd, parser=parser)
                keep = None
                title = None
                author = None
                for ts in xml.iter("{*}titleStmt"):
                    for t in ts.iter("{*}title"):
                        if t.text:
                            title = t.text


                    for a in ts.iter("{*}author"):
                        if a.text and not re.match(r"^\s*$", a.text):
                            author = a.text

                date = author_titles.get((author, title), titles.get(title, authors.get(author, None)))

                if date:
                    content = []
                    for div in xml.iter("{*}div"):
                        for k, v in div.attrib.items():
                            if v == "lat" and div.attrib.get("type") != "translation":
                                content.append(" ".join(div.xpath(".//text()")))
                    if len(content) > 0:
                        content = " ".join(content)
                        ofd.write(
                            json.dumps(
                                {
                                    "file" : fname,
                                    "author" : author,
                                    "title" : title,
                                    "inferred_year" : date,
                                    "content" : content
                                }
                            ) + "\n"
                        )
