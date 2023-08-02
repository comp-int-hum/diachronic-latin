import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# actual variable and environment objects
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("FOLDS", "", 5),
    ("N", "", 3),
    ("RANDOM_SEED", "", 0),
    ("TRAIN_PROPORTION", "", 0.8),
    ("DEV_PROPORTION", "", 0.1),
    ("TEST_PROPORTION", "", 0.1),    
    ("HATHITRUST_ROOT", "", "/export/large_corpora/hathi_trust"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_full_20211001.txt.gz"),
)


env = Environment(
    variables=vars,
    ENV=os.environ,
    TARFLAGS="-c -z",
    TARSUFFIX=".tgz",
    tools=[steamroller.generate],
    BUILDERS={
        "FilterHathiTrust" : Builder(
            action="python scripts/filter_hathitrust.py --output ${TARGETS[0]} --hathitrust_index ${HATHITRUST_INDEX}"
        ),
        "BuildDocumentList" : Builder(
            action="python scripts/build_document_list.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "HydrateDocumentList" : Builder(
            action="python scripts/hydrate_document_list.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT}"
        ),
        "FilterAndNarrowDocuments" : Builder(
            action="python scripts/filter_and_narrow_documents.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        )
    }
)

# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)

# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line

# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")

latin_documents = env.FilterHathiTrust(
    "work/latin_documents.tsv.gz",
    []
)

relevant_document_list = env.BuildDocumentList(
    "work/relevant_latin_documents.jsonl.gz",
    latin_documents
)

hydrated_document_list = env.HydrateDocumentList(
    "work/hydrated_documents.jsonl.gz",
    relevant_document_list
)

final_documents = env.FilterAndNarrowDocuments(
    "work/final_documents.jsonl.gz",
    hydrated_document_list
)
