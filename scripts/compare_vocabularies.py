import argparse
import os.path
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from pairtree import PairtreeStorageFactory
import re
import zipfile
from xml.etree import ElementTree as et
from bs4 import BeautifulSoup

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_htid", dest="input_htid")
    parser.add_argument("--input_perseus", dest="input_perseus")
    parser.add_argument("--perseus_root", dest="perseus_root")
    parser.add_argument("--hathitrust_root", dest="hathitrust_root")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(os.path.join(args.perseus_root, args.input_perseus), "rt") as ifd:
        soup = BeautifulSoup(ifd.read(), 'xml')
    perseus_vocab = {}
    for i in range(1, 7):
        b = soup.find(type="Book", n=str(i))
        for line in b.find_all("l"):
            for tok in line.text.split():
                perseus_vocab[tok] = perseus_vocab.get(tok, 0) + 1
    print(len(perseus_vocab), sum(perseus_vocab.values()))
    
    psf = PairtreeStorageFactory()
    subcollection, ident = re.match(r"^([^\.]+)\.(.*)$", args.input_htid).groups()
    store = psf.get_store(
        store_dir=os.path.join(
            args.hathitrust_root,
            subcollection
        )
    )
    obj = store.get_object(ident, create_if_doesnt_exist=False)
    pages = []
    for subpath in obj.list_parts():
        for fname in obj.list_parts(subpath):
            if fname.endswith("zip"):
                with zipfile.ZipFile(
                        obj.get_bytestream(
                            "{}/{}".format(subpath, fname),
                            streamable=True
                        )
                ) as izf:                            
                    for page in sorted(izf.namelist()):
                        if page.endswith("txt"):
                            txt = izf.read(page).decode("utf-8")
                            pages.append(txt)
    # books 1-6
    text = "\n".join(pages[60:289])
    ht_vocab = {}
    for tok in text.split():
        ht_vocab[tok] = ht_vocab.get(tok, 0) + 1
    print(len(ht_vocab), sum(ht_vocab.values()))
