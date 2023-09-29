from dataclasses import dataclass, field, asdict
from boltons.cacheutils import cachedproperty
from typing import List, Type
import re
import os.path
import os
from zipfile import ZipFile
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
    @classmethod
    def get_nlp(cls, language: str, treebank = None):
        if language in cls.nlps:
            return cls.nlps[language]
        else:
            nlp = cls(language, treebank, interactive=False)
            cls.nlps[language] = nlp
            return nlp

@dataclass
class CustomStanzaProcess(LatinStanzaProcess):
    @cachedproperty
    def algorithm(self):
        return CustomStanzaWrapper.get_nlp(language=self.language)

@dataclass
class CustomPipeline(Pipeline):
    description: str = "Custom pipeline for Latin w/o embedding or lexicon step"
    language: Language = get_lang("lat")
    processes: List[Type[Process]] = field(
        default_factory=lambda: [
            LatinNormalizeProcess,
            CustomStanzaProcess,
        ]
    )

def serialize(item):
    if isinstance(item, Word):
        return asdict(item)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--perseus_corpus", dest="perseus_corpus", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    perseus_vocab = {}
    with ZipFile(args.perseus_corpus, "r") as zifd:
        for name in zifd.namelist():
            if re.match(r".*lat\.xml", name):
                with zifd.open(name, "r") as ifd:
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
