import argparse
import logging
import json
import gzip
import re
import math
import random


logger = logging.getLogger("format_for_dtm.py")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("--text_output", dest="text_output", help="Output file")
    parser.add_argument("--time_output", dest="time_output", help="Output file")
    parser.add_argument("--vocab_output", dest="vocab_output", help="Output file")
    parser.add_argument("--metadata_output", dest="metadata_output", help="Output file")
    parser.add_argument("--tokens_per_chunk", dest="tokens_per_chunk", default=200, type=int)
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int)
    parser.add_argument("--window_count", dest="window_count", default=10, type=int)
    parser.add_argument("--max_chunk_proportion", dest="max_chunk_proportion", default=0.1, type=float)
    parser.add_argument("--min_token_proportion", dest="min_token_proportion", default=0.00001, type=float)
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", default=False)
    parser.add_argument("--max_chunks_per_window", dest="max_chunks_per_window", default=300, type=int)
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["INFO", "WARN", "DEBUG", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if args.random_seed:
        random.seed(args.random_seed)
    
    logger.info("First pass to gather chunk-counts for down-sampling")
    docs = []
    tok_counts = {}
    tok_chunk_counts = {}
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            js = json.loads(line)
            toks = []
            for sent in js["content"]:
                toks += sent
            span_count = math.ceil(len(toks) / args.tokens_per_chunk)
            date = js["inferred_year"]
            for j in range(span_count - 1):
                toks_in_chunk = set()
                for tok in toks[j * args.tokens_per_chunk : (j + 1) * args.tokens_per_chunk]:
                    tok_counts[tok] = tok_counts.get(tok, 0) + 1
                    toks_in_chunk.add(tok)
                for tok in toks_in_chunk:
                    tok_chunk_counts[tok] = tok_chunk_counts.get(tok, 0) + 1
            docs.append((int(date), span_count, i))
    docs = list(sorted(docs))
    start = docs[0][0]
    end = docs[-1][0]
    length = end - start
    years_per_window = length / args.window_count
    total_chunk_count = sum([x for _, x, _ in docs])
    total_token_count = sum(tok_counts.values())
    chunkcount_keep = set([t for t, cc in tok_chunk_counts.items() if cc / total_chunk_count <= args.max_chunk_proportion])
    count_keep = set([t for t, c in tok_counts.items() if (c / total_token_count) >= args.min_token_proportion])
    keep_tokens = chunkcount_keep.intersection(count_keep)
    
    logger.info(
        """
        Years per window: %.3f
        Total document count: %d
        Total chunk count: %d
        Unique token count: %d
        Total token count: %d
        Filtered vocab size: %d
        """,
        years_per_window,
        len(docs),
        total_chunk_count,
        len(tok_counts),
        total_token_count,
        len(keep_tokens)
    )

    logger.info("Assembling and down-sampling temporal windows")

    windows = {}
    cur_max = start + years_per_window
    cur_window = 0
    for doc in docs:
        if doc[0] > cur_max:            
            cur_window += 1
            cur_max += years_per_window
        windows[cur_window] = windows.get(cur_window, []) + [(doc[2], i) for i in range(doc[1])]

    keep_docs = set()
    for bid, entries in windows.items():
        random.shuffle(entries)
        for d, c in entries[:args.max_chunks_per_window]:
            keep_docs.add((d, c))
                
    logger.info("Second pass to load and format data for DTM")

    #total_token_count = 0
    #total_doc_count = 0
    #id2doccount = {}
    #id2count = {}
    tok2id = {}
    docs = []    
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            js = json.loads(line)
            toks = []
            for sent in js["content"]:
                toks += sent
            span_count = math.ceil(len(toks) / args.tokens_per_chunk)
            date = js["inferred_year"]
            chunks = []
            for j in range(span_count - 1):
                if (i, j) not in keep_docs:
                    continue
                #total_doc_count += 1
                subtoks = {}
                for tok in toks[j * args.tokens_per_chunk : (j + 1) * args.tokens_per_chunk]:
                    if tok not in keep_tokens:
                        continue
                    tok2id[tok] = tok2id.get(tok, len(tok2id))
                    tokid = tok2id.get(tok)
                    subtoks[tokid] = subtoks.get(tokid, 0) + 1
                chunks.append((j, subtoks))
            docs.append((date, {k : v for k, v in js.items() if k != "content"}, chunks))
    docs = list(sorted(docs, key=lambda x : x[0]))
    
    cur_max = start + years_per_window
    with open(args.text_output, "wt") as text_ofd, open(args.time_output, "wt") as time_ofd, gzip.open(args.metadata_output, "wt") as metadata_ofd:
        time_ofd.write("{}\n".format(args.window_count))
        cur_count = 0
        for doc in docs:
            if doc[0] > cur_max:
                time_ofd.write("{}\n".format(cur_count))
                cur_count = 0
                cur_max += years_per_window
            for chunk in doc[2]:
                cur_count += 1
                text_ofd.write("{} {}\n".format(len(chunk[1]), " ".join(["{}:{}".format(k, v) for k, v in chunk[1].items()])))
                metadata_ofd.write(json.dumps(doc[1]) + "\n")
        if cur_count > 0:
            time_ofd.write("{}\n".format(cur_count))
    with gzip.open(args.vocab_output, "wt") as ofd:
        ofd.write(json.dumps(tok2id, indent=4))

