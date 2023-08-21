from dataclasses import dataclass, field, asdict
from typing import List, Type
import re
import os.path
import os
import argparse
import json
import gzip
import logging
from glob import glob
from bs4 import BeautifulSoup
from cltk.nlp import NLP
from cltk.dependency.processes import LatinStanzaProcess
from cltk.dependency.stanza import StanzaWrapper
from cltk.core.data_types import Pipeline, Process, Language, Doc, Word, Sentence
from cltk.languages.utils import get_lang
from cltk.alphabet.processes import LatinNormalizeProcess

logger = logging.getLogger("extract_features.py")

class CustomStanzaWrapper(StanzaWrapper):
    def _load_pipeline(self):
        models_dir = os.path.expanduser(
            "~/stanza_resources/"
        )  # TODO: Mv this a self. var or maybe even global
        processors = "tokenize,mwt,pos,lemma,depparse"
        lemma_use_identity = False
        if self.language == "fro":
            processors = "tokenize,pos,lemma,depparse"
            lemma_use_identity = True
        if self.language in ["chu", "got", "grc", "lzh"]:
            # Note: MWT not available for several languages
            processors = "tokenize,pos,lemma,depparse"
        nlp = stanza.Pipeline(
            lang=self.stanza_code,
            dir=models_dir,
            package=self.treebank,
            processors=processors,  # these are the default processors
            logging_level=self.stanza_debug_level,
            use_gpu=True,  # default, won't fail if GPU not present
            kwargs={"depparse_batch_size" : 20000},
            lemma_use_identity=lemma_use_identity,
        )
        return nlp

@dataclass
class CustomStanzaProcess(LatinStanzaProcess):
    def algorithm(self):
        return CustomStanzaWrapper.get_nlp(language=self.language)

@dataclass
class CustomPipeline(Pipeline):
    description: str = "Custom pipeline for Latin w/o embedding or lexicon step"
    language: Language = get_lang("lat")
    processes: List[Type[Process]] = field(
        default_factory=lambda: [
            LatinNormalizeProcess,
            LatinStanzaProcess,
        ]
    )

def serialize(item):
    if isinstance(item, Word):
        return asdict(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--perseus_root", dest="perseus_root", default="/export/data/classics/perseus")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    perseus_vocab = {}
    for fname in glob(os.path.join(args.perseus_root, "*/*/*lat*")):
        with open(fname, "rt") as ifd:
            soup = BeautifulSoup(ifd.read(), 'xml')
            for text in soup.find_all("text"):
                for tok in re.split(r"\s+", text.get_text().lower()):
                    perseus_vocab[tok] = perseus_vocab.get(tok, 0) + 1
    perseus_vocab = {k : v for k, v in perseus_vocab.items() if v > 3}
    
    nlp = NLP(language="lat", custom_pipeline=CustomPipeline(), suppress_banner=True)

    try:
        with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
            for i, line in enumerate(ifd):
                logger.info("Processing document #%d", i + 1)
                j = json.loads(line)
                text = " ".join([a if (a in perseus_vocab or b not in perseus_vocab) else b for (a, b) in [(t.lower(), t.lower().replace("f", "s")) for t in re.split(r"\s+", j["content"])]])
                j["content"] = [s.words for s in nlp.analyze(text).sentences]
                logger.info("Parsed %s sentences", len(j["content"]))
                ofd.write(json.dumps(j, default=serialize) + "\n")
    except Exception as e:
        os.remove(args.output)
        raise e
