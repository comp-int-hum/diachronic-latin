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
    ("PERSEUS_ROOT", "", "/export/data/classics/perseus"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_full_20211001.txt.gz"),
    ("DTM_ROOT", "Path to the (compiled) dynamic topic modeling repository", "~/dtm"),
    ("TOKENS_PER_CHUNK", "", 200),
    ("WINDOW_COUNT", "", 20),
    ("RANDOM_SEED", "", 0),
    ("MIN_TOKEN_PROPORTION", "", 0.00001),
    ("MAX_CHUNK_PROPORTION", "", 0.2),
    ("MAX_CHUNKS_PER_WINDOW", "", 500),
    ("TOPIC_COUNT", "", 20),
    ("USE_PRECOMPUTED_FEATURES", "If set, the file 'work/features.jsonl.gz' should already exist (e.g. from an earlier invocation of the build, or copied from another location)", False),
    ("USE_PRETRAINED_TOPIC_MODELS", "If set, the various 'dtm_model*.tgz' files should exist under work/", False),    
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
        "PerformLID" : Builder(
            action="python scripts/perform_lid.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),        
        "FilterAndNarrowDocuments" : Builder(
            action="python scripts/filter_and_narrow_documents.py --lid ${SOURCES[0]} --content ${SOURCES[1]} --output ${TARGETS[0]}"
        ),
        "ExtractFeatures" : Builder(
            action="python scripts/extract_features.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "FormatForDTM" : Builder(
            action="python scripts/format_for_dtm.py --input ${SOURCES[0]} --text_output ${TARGETS[0]} --time_output ${TARGETS[1]} --vocab_output ${TARGETS[2]} --metadata_output ${TARGETS[3]} --tokens_per_chunk ${TOKENS_PER_CHUNK} --window_count ${WINDOW_COUNT} --random_seed ${RANDOM_SEED} --min_token_proportion ${MIN_TOKEN_PROPORTION} --max_chunk_proportion ${MAX_CHUNK_PROPORTION}  --lowercase --max_chunks_per_window ${MAX_CHUNKS_PER_WINDOW}"
        ),
        "TrainDTM" : Builder(
            action="python scripts/train_dtm.py --text_input ${SOURCES[0]} --time_input ${SOURCES[1]} --dtm_path ${DTM_ROOT} --output ${TARGETS[0]} --topic_count ${TOPIC_COUNT}"
        ),
        "InspectDTM" : Builder(
            action="python scripts/inspect_dtm.py --model ${SOURCES[0]} --vocab ${SOURCES[1]} --metadata ${SOURCES[2]} --text ${SOURCES[3]} --time ${SOURCES[4]} --docs ${SOURCES[5]} --output ${TARGETS[0]}"
        ),
        "CompareVocabularies" : Builder(
            action="python scripts/compare_vocabularies.py --input_htid ${INPUT_HTID} --input_perseus ${INPUT_PERSEUS} --hathitrust_root ${HATHITRUST_ROOT} --output ${TARGETS[0]} --perseus_root ${PERSEUS_ROOT}"
        ),
        "SplitJSONL" : Builder(
            action="python scripts/split_jsonl.py --input ${SOURCES[0]} --outputs ${TARGETS}"
        ),
        "CombineJSONL" : Builder(
            action="python scripts/combine_jsonl.py --inputs ${SOURCES} --output ${TARGETS[0]}"
        ),
        "FilterUnknownWords" : Builder(
            action="python scripts/filter_unknown_words.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "CombineFeatures" : Builder(
            action="python scripts/combine_features.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
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

if env["USE_PRECOMPUTED_FEATURES"]:

    cltk_features = env.File("work/features.jsonl.gz")
    
else:

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

    lid_output = env.PerformLID(
       "work/lid_output.jsonl.gz",
       hydrated_document_list
    )

    final_documents = env.FilterAndNarrowDocuments(
        "work/final_documents.jsonl.gz",
        [lid_output, hydrated_document_list]
    )

    feat_files = []
    for j, subfile in enumerate(
            env.SplitJSONL(
                ["work/final_documents_split_{}.jsonl.gz".format(i) for i in range(1, 21)],
                final_documents
            )
    ):
        feat_files.append(
            env.ExtractFeatures(
                "work/features_split_{}.jsonl.gz".format(j + 1),
                subfile
            )
        )

    cltk_features = env.CombineJSONL(
        "work/features.jsonl.gz",
        feat_files
    )

text_for_dtm = env.FilterUnknownWords(
    "work/text_for_dtm.jsonl.gz",
    cltk_features
)

if env["USE_PRETRAINED_TOPIC_MODELS"]:

    model = env.File("work/dtm_model.tgz")
    
else:
    
    dtm_text_input, dtm_time_input, vocab, metadata = env.FormatForDTM(
        [
            "work/dtm_input-mult.dat",
            "work/dtm_input-seq.dat",
            "work/dtm_vocab.json.gz",
            "work/dtm_doc_metadata.jsonl.gz"
        ],
        text_for_dtm
    )

    topic_model = env.TrainDTM(
        "work/dtm_model.tgz",
        [
            dtm_text_input,
            dtm_time_input
        ]
    )

# topic_model_features = env.ApplyDTM(
#     "work/dtm_features.jsonl.gz",
#     [
#         topic_model,
#         text_for_dtm
#     ]
# )

summary = env.InspectDTM(
    "work/dtm_summary.txt",
    [
        topic_model,
        vocab,
        metadata,
        dtm_text_input,
        dtm_time_input,
        text_for_dtm,
    ]
)

# vocab_comparison = env.CompareVocabularies(
#     "work/vocab_comparison.txt",
#     [],
#     INPUT_HTID="njp.32101058331438",
#     INPUT_PERSEUS="Vergil/opensource/verg.a_lat.xml"
# )

# final_features = env.CombineFeatures(
#     "work/final_features.jsonl.gz",
#     [cltk_features, topic_model_features]
# )
