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
    ("FOLDS", "", 5),
    ("N", "", 3),
    ("RANDOM_SEED", "", 0),
    ("TRAIN_PROPORTION", "", 0.8),
    ("DEV_PROPORTION", "", 0.1),
    ("TEST_PROPORTION", "", 0.1),    
    ("HATHITRUST_ROOT", "", "~/corpora/hathi_trust"),
    ("PERSEUS_ROOT", "", "~/corpora/perseus"),
    ("PERSEUS_CORPUS", "", "~/corpora/perseus_classics.zip"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_full_20211001.txt.gz"),
    ("TOKENS_PER_CHUNK", "", 200),
    ("WINDOW_COUNT", "", 20),
    ("RANDOM_SEED", "", 0),
    ("MIN_TOKEN_PROPORTION", "", 0.00001),
    ("MAX_CHUNK_PROPORTION", "", 0.2),
    #("TOPIC_COUNT", "", 20),
    ("USE_PRECOMPUTED_FEATURES", "If set, the file 'work/features.jsonl.gz' should already exist (e.g. from an earlier invocation of the build, or copied from another location)", False),
    ("USE_PRETRAINED_TOPIC_MODELS", "If set, the various 'dtm_model*.tgz' files should exist under work/", False),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),


    #("MAX_SUBDOC_LENGTHS", "", [500]), #[200, 500, 1000, 2000]),
    #("MIN_WORD_OCCURRENCE", "", 10),
    #("MAX_WORD_PROPORTION", "", 0.7),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),
    ("TOPIC_COUNT", "", 50),
    #("EPOCHS", "", 500),
    #("LIMIT_DOCS", "", 2020),
    #("CUDA_DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 1000),
    ("WINDOW_SIZE", "", 75),
    #("LEARNING_RATE", "", 0.0001),
    #("DETM_PATH", "", "../DETM"),
    ("USE_PRECOMPUTED_FEATURES", "If set, the file 'work/features.jsonl.gz' should already exist", False),
    ("USE_PRETRAINED_TOPIC_MODELS", "", False),
    ("USE_PRETRAINED_EMBEDDINGS", "", False),
    ("PREFIXES_TO_PRESERVE", "", ["fig", "fing", "fact", "effig", "fict", "nov", "color", "curs", "magn", "form", "multi", "morph", "eid", "schem", "typ", "plas", "mege", "kines", "chrom", "exempl", "statu", "imag", "spec", "simula", "membr", "primord", "princip", "corp", "element", "sem"]),
    ("AUTHOR_TARGETS", "", ["Varro", "Cicero", "Lucretius"]),
    ("WORD_SIMILARITY_TARGETS", "", ["figura", "imago", "corpus"]),
    ("MIN_TIME", "", -250),
    ("MAX_TIME", "", 500)
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
            action="python scripts/build_document_list.py --input ${SOURCES[0]} --output ${TARGETS[0]} --min_year ${MIN_TIME} --max_year ${MAX_TIME}"
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
            action="python scripts/extract_features.py --input ${SOURCES[0]} --output ${TARGETS[0]} --perseus_corpus ${PERSEUS_CORPUS}"
        ),
        "FormatForDTM" : Builder(
            action="python scripts/format_for_dtm.py --input ${SOURCES[0]} --text_output ${TARGETS[0]} --time_output ${TARGETS[1]} --vocab_output ${TARGETS[2]} --metadata_output ${TARGETS[3]} --tokens_per_chunk ${TOKENS_PER_CHUNK} --window_count ${WINDOW_COUNT} --random_seed ${RANDOM_SEED} --min_token_proportion ${MIN_TOKEN_PROPORTION} --max_chunk_proportion ${MAX_CHUNK_PROPORTION}  --lowercase --max_chunks_per_window ${MAX_CHUNKS_PER_WINDOW}"
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
        ),
        "MarshallPerseus" : Builder(
            action="python scripts/marshall_perseus.py --perseus_root ${PERSEUS_ROOT} --date_ranges data/date_ranges.jsonl --output ${TARGETS[0]}"
        ),
        "TrainEmbeddings" : Builder(
            action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "TrainDETM" : Builder(
            action="python scripts/train_detm.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --output ${TARGETS[0]} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION} ${'--min_time ' + str(MIN_TIME) if MIN_TIME else ''} ${'--max_time ' + str(MAX_TIME) if MAX_TIME else ''}"
        ),
        "ApplyDETM" : Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --output ${TARGETS[0]}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --model ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS}"
        ),
        "GenerateTopicsSummaryTable" : Builder(
            action="python scripts/generate_topics_summary_table.py --model ${SOURCES[0]} --output ${TARGETS[0]} --top_words ${TOP_TOPIC_WORDS}"
        ),
                "CollectParsingStatistics" : Builder(
            action="python scripts/collect_parsing_statistics.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "CollectTopicStatistics" : Builder(
            action="python scripts/collect_topic_statistics.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "CreateMatrices" : Builder(
            action="python scripts/create_matrices.py --parse_stats ${SOURCES[0]} --topic_annotations ${SOURCES[1]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE} --min_time ${MIN_TIME}"
        ),
        "CreateFigures" : Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --image ${TARGETS[0]} --latex ${TARGETS[1]}"
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

perseus_documents = env.MarshallPerseus(
    "work/perseus.jsonl.gz",
    []
)

lid_output = env.PerformLID(
    "work/lid_output.jsonl.gz",
    hydrated_document_list,
    GRID_MEMORY="16G"
)

htc_documents = env.FilterAndNarrowDocuments(
    "work/final_documents.jsonl.gz",
    [lid_output, hydrated_document_list]
)

for name, docs in [("perseus", perseus_documents), ("htc", htc_documents)][:1]:
    feat_files = []
    for j, subfile in enumerate(
            env.SplitJSONL(
                ["work/final_documents_split_{}_{}.jsonl.gz".format(name, i) for i in range(1, 51)],
                docs
            )
    ):
        feat_files.append(
           env.ExtractFeatures(
               "work/features_split_{}_{}.jsonl.gz".format(name, j + 1),
               subfile,
               GRID_MEMORY="32G"               
           )
        )

    cltk_features = env.CombineJSONL(
        "work/features_{}.jsonl.gz".format(name),
        feat_files
    )

    text_for_detm = env.FilterUnknownWords(
        "work/text_for_detm_{}.jsonl.gz".format(name),
        cltk_features
    )

    parse_stats = env.CollectParsingStatistics(
        "work/parsing_stats_{}.jsonl.gz".format(name),
        cltk_features
    )

    embeddings = env.TrainEmbeddings(
        "work/embeddings_{}.bin".format(name),
        text_for_detm,
        GRID_MEMORY="32G"
    )

    topic_model = env.TrainDETM(
        "work/detm_model_${MAX_SUBDOC_LENGTH}_%s.bin.gz" % (name),
        [
            text_for_detm,
            embeddings
        ],
        BATCH_SIZE=2000,
        EPOCHS=1000,
        MIN_WORD_OCCURRENCE=1,
        MAX_WORD_PROPORTION=0.7,
        #WINDOW_SIZE=100,
        LEARNING_RATE=0.0008*20,
        CUDA_DEVICE="cuda",
        MAX_SUBDOC_LENGTH=500,
        GRID_MEMORY="32G",
        GRID_QUEUE="a100",
        GRID_GPU_COUNT="1",
        GRID_ACCOUNT="tlippin1_gpu"
    )

    word_similarity_table = env.GenerateWordSimilarityTable(
        "work/word_similarity_${MAX_SUBDOC_LENGTH}_%s.tex" % (name),
        [topic_model],
        WORD_SIMILARITY_TARGETS=["figura", "imago", "fortuna", "praefectus", "corpus", "bellum"],
        MAX_SUBDOC_LENGTH=500,
        TOP_NEIGHBORS=5
    )

    topics_table = env.GenerateTopicsSummaryTable(
        "work/topics_summary_${MAX_SUBDOC_LENGTH}_%s.tex" % (name),
        topic_model,
        MAX_SUBDOC_LENGTH=500,        
    )

    # labeled = env.ApplyDETM(
    #     "work/labeled_${MAX_SUBDOC_LENGTH}_%s.jsonl.gz" % (name),
    #     [topic_model, text_for_detm],
    #     MAX_SUBDOC_LENGTH=500,
    #     GRID_MEMORY="32G",
    #     #GRID_QUEUE="a100",
    #     #GRID_GPU_COUNT="1",
    #     #GRID_ACCOUNT="tlippin1_gpu"        
    # )

    # matrices = env.CreateMatrices(
    #     "work/matrices_${MAX_SUBDOC_LENGTH}_%s.pkl.gz" % (name),
    #     [
    #         parse_stats,
    #         labeled
    #     ],
    #     MAX_SUBDOC_LENGTH=500,
    #     GRID_MEMORY="64G"
    # )
    
    # figures = env.CreateFigures(
    #     [
    #         "work/plots_${MAX_SUBDOC_LENGTH}_%s.png" % (name),
    #         "work/tables_${MAX_SUBDOC_LENGTH}_%s.tex" % (name)
    #     ],
    #     matrices,
    #     MAX_SUBDOC_LENGTH=500,
    #     WINDOW_SIZE=100
    # )


    
    # if env["USE_PRETRAINED_TOPIC_MODELS"]:

    #     model = env.File("work/dtm_model_{}.tgz".format(name))

    # else:

    #     dtm_text_input, dtm_time_input, vocab, metadata = env.FormatForDTM(
    #         [
    #             "work/dtm_input-mult.dat",
    #             "work/dtm_input-seq.dat",
    #             "work/dtm_vocab.json.gz",
    #             "work/dtm_doc_metadata.jsonl.gz"
    #         ],
    #         text_for_dtm
    #     )

    #     topic_model = env.TrainDTM(
    #         "work/dtm_model.tgz",
    #         [
    #             dtm_text_input,
    #             dtm_time_input
    #         ]
    #     )

    # topic_model_features = env.ApplyDTM(
    #     "work/dtm_features.jsonl.gz",
    #     [
    #         topic_model,
    #         text_for_dtm
    #     ]
    # )

    # summary = env.InspectDTM(
    #     "work/dtm_summary.txt",
    #     [
    #         topic_model,
    #         vocab,
    #         metadata,
    #         dtm_text_input,
    #         dtm_time_input,
    #         text_for_dtm,
    #     ]
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
