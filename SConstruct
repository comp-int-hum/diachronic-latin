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
from steamroller import Environment

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# actual variable and environment objects
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("RANDOM_SEED", "", 0),
    ("HATHITRUST_ROOT", "", "~/corpora/hathi_trust"),
    ("PERSEUS_CORPUS", "", "~/corpora/perseus_classics.zip"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_full_20211001.txt.gz"),
    ("MAX_SUBDOC_LENGTHS", "", [500]), #[200, 500, 1000, 2000]),
    ("MIN_WORD_OCCURRENCE", "", 10),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),
    ("TOPIC_COUNT", "", 50),
    ("EPOCHS", "", 500),
    ("LIMIT_DOCS", "", 2020),
    ("CUDA_DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 1000),
    ("WINDOW_SIZE", "", 50),
    ("LEARNING_RATE", "", 0.0001),
    ("DETM_PATH", "", "../DETM"),
    ("USE_PRECOMPUTED_FEATURES", "If set, the file 'work/features.jsonl.gz' should already exist", False),
    ("USE_PRETRAINED_TOPIC_MODELS", "", False),
    ("USE_PRETRAINED_EMBEDDINGS", "", False),
    ("PREFIXES_TO_PRESERVE", "", ["fig", "fing", "fact", "effig", "fict", "nov", "color", "curs", "magn", "form", "multi", "morph", "eid", "schem", "typ", "plas", "mege", "kines", "chrom", "exempl", "statu", "imag", "spec", "simula", "membr", "primord", "princip", "corp", "element", "sem"]),
    ("AUTHOR_TARGETS", "", ["Varro", "Cicero", "Lucretius"]),
    ("WORD_SIMILARITY_TARGETS", "", ["figura", "imago", "corpus"]),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    TARFLAGS="-c -z",
    TARSUFFIX=".tgz",
    tools=[],
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
        "TrainEmbeddings" : Builder(
            action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "TrainDETM" : Builder(
            action="python scripts/train_detm.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --output ${TARGETS[0]} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION}"
        ),
        "ApplyDETM" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --output ${TARGETS[0]}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --model ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS}"
        ),
        "GenerateTopicsSummaryTable" : Builder(
            action="python scripts/generate_topics_summary_table.py --model ${SOURCES[0]} --output ${TARGETS[0]} --top_words ${TOP_TOPIC_WORDS}"
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
            action="python scripts/filter_unknown_words.py --input ${SOURCES[0]} --output ${TARGETS[0]} --prefixes_to_preserve ${PREFIXES_TO_PRESERVE}"
        ),
        "CombineFeatures" : Builder(
            action="python scripts/combine_features.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "FormatDETMInput" : Builder(
            action="python scripts/format_detm_input.py --data_path ${DATA_PATH} --train_output ${TARGETS[0]} --val_output ${TARGETS[1]} --test_output ${TARGETS[2]}"
        ),
        "CollectParsingStatistics" : Builder(
            action="python scripts/collect_parsing_statistics.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "CollectTopicStatistics" : Builder(
            action="python scripts/collect_topic_statistics.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "CreateMatrices" : Builder(
            action="python scripts/create_matrices.py --parse_stats ${SOURCES[0]} --topic_annotations ${SOURCES[1]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE}"
        ),
        "CreateFigures" : Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --plots ${TARGETS[0]} --tables ${TARGETS[1]}"
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

text_for_detm = env.File("text_for_detm.jsonl.gz")

#env.FilterUnknownWords(
#    "work/text_for_detm.jsonl.gz",
#    cltk_features
#)

parse_stats = env.CollectParsingStatistics(
    "work/parsing_stats.jsonl.gz",
    cltk_features
)

if env["USE_PRETRAINED_EMBEDDINGS"]:
    embeddings = env.File("work/embeddings.bin")
else:
    embeddings = env.TrainEmbeddings(
        "work/embeddings.bin",
        text_for_detm
    )

for msl in env["MAX_SUBDOC_LENGTHS"]:

    if env["USE_PRETRAINED_TOPIC_MODELS"]:
        topic_model = env.File("work/detm_model_{}.bin.gz".format(msl))
    else:
        topic_model = env.TrainDETM(
            "work/detm_model_${MAX_SUBDOC_LENGTH}.bin.gz",
            [
                text_for_detm,
                embeddings
            ],
            BATCH_SIZE=2000,
            EPOCHS=1000,
            MIN_WORD_OCCURRENCE=1,
            MAX_WORD_PROPORTION=0.7,
            WINDOW_SIZE=100,
            LEARNING_RATE=0.0008*20,
            CUDA_DEVICE="cuda:1",
            MAX_SUBDOC_LENGTH=msl
        )

    word_similarity_table = env.GenerateWordSimilarityTable(
        "work/word_similarity_${MAX_SUBDOC_LENGTH}.tex",
        [topic_model],
        WORD_SIMILARITY_TARGETS=["figura", "imago", "fortuna", "praefectus", "candidatus"],
        MAX_SUBDOC_LENGTH=msl,
        TOP_NEIGHBORS=5
    )

    topics_table = env.GenerateTopicsSummaryTable(
        "work/topics_summary_${MAX_SUBDOC_LENGTH}.tex",
        topic_model,
        MAX_SUBDOC_LENGTH=msl,        
    )

    labeled = env.ApplyDETM(
        "work/labeled_${MAX_SUBDOC_LENGTH}.jsonl.gz",
        [topic_model, text_for_detm],
        MAX_SUBDOC_LENGTH=msl
    )
    
    # topic_stats = env.CollectTopicStatistics(
    #     "work/topic_stats_${MAX_SUBDOC_LENGTH}.json.gz",
    #     labeled,
    #     MAX_SUBDOC_LENGTH=msl
    # )

    matrices = env.CreateMatrices(
        "work/matrices_${MAX_SUBDOC_LENGTH}.pkl.gz",
        [
            parse_stats,
            labeled
        ],
        MAX_SUBDOC_LENGTH=msl,
        WINDOW_SIZE=100
    )
    
    figures = env.CreateFigures(
        [
            "work/plots_${MAX_SUBDOC_LENGTH}.png",
            "work/tables_${MAX_SUBDOC_LENGTH}.tex"
        ],
        matrices,
        MAX_SUBDOC_LENGTH=msl,
        WINDOW_SIZE=100
    )
    
# acl_train, acl_val, acl_test = env.FormatDETMInput(
#     [
#         "work/acl_train.jsonl.gz",
#         "work/acl_val.jsonl.gz",
#         "work/acl_test.jsonl.gz",
#     ],
#     [],
#     DATA_PATH="${DETM_PATH}/data_acl_largev/min_df_10",
# )

# acl_topic_model = env.TrainDETM(
#     "work/acl_detm_model.bin.gz",
#     [
#         acl_train,
#         "${DETM_PATH}/embeddings/acl/skipgram_emb_300d.txt",
#         acl_val,
#     ],
#     BATCH_SIZE=1000,
#     EPOCHS=30,
#     MIN_WORD_OCCURRENCE=0,
#     MAX_WORD_PROPORTION=1.0,
#     WINDOW_SIZE=1,
#     LEARNING_RATE=0.0008*10,
#     CUDA_DEVICE="cuda:1",
#     MAX_SUBDOC_LENGTH=100000    
# )
    
# acl_word_similarity_table = env.GenerateWordSimilarityTable(
#     "work/acl_word_similarity.tex",
#     [acl_topic_model],
#     WORD_SIMILARITY_TARGETS=["neural", "parse", "semantics", "translation", "memory"]
# )

# acl_topics_table = env.GenerateTopicsSummaryTable(
#    "work/acl_topics_summary.tex",
#    acl_topic_model
# )

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
