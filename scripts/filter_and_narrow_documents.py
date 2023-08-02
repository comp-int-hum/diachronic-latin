import argparse
import gzip
import re
import json
import math
import fasttext
from huggingface_hub import hf_hub_download


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--tokens_per_chunk", dest="tokens_per_chunk", type=int, default=200, help="")
    args = parser.parse_args()

    model_path = hf_hub_download(repo_id="arc-r/fasttext-language-identification", filename="lid.176.bin")
    model = fasttext.load_model(model_path)
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            toks = re.split(r"\s+", j["content"])
            span_count = math.ceil(len(toks) / args.tokens_per_chunk)
            chunks = [" ".join(toks[i * args.tokens_per_chunk : (i + 1) * args.tokens_per_chunk]) for i in range(span_count)]
            labels, probs = model.predict(chunks, k=3)
            final_labels = []
            for chunk, l, p in zip(chunks, labels, probs):                
                if l[0] != "__label__la" and len(chunk) > 0:                    
                    digit_prop = len(re.sub(r"[^0-9]", "", chunk)) / len(chunk)
                    punct_prop = len(re.sub(r"[^{}()-~.';[\]]", "", chunk)) / len(chunk)
                    if digit_prop > 0.25:
                        label = "noise"
                    else:
                        label = l[0].split("_")[-1]
                else:
                    label = "la"
                final_labels.append(label)
            ofd.write(json.dumps(dict([(k, v) for k, v in j.items() if k != "content"] + [("labels", final_labels)])) + "\n")

