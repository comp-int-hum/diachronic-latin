import argparse
import subprocess
import os.path
import tempfile
import os
import re
import tarfile
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtm_path", dest="dtm_path", help="Path to root of DTM repo (that has been compiled per its README)")
    parser.add_argument("--text_input", dest="text_input", help="Input file")
    parser.add_argument("--time_input", dest="time_input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--topic_count", dest="topic_count", default=10, type=int)
    parser.add_argument("--max_lda_em_iters", dest="max_lda_em_iter", default=10, type=int)
    parser.add_argument("--min_lda_seq_iter", dest="min_lda_seq_iter", default=6, type=int)
    parser.add_argument("--max_lda_seq_iter", dest="max_lda_seq_iter", default=10, type=int)
    parser.add_argument("--alpha", dest="alpha", default=0.01, type=float)
    parser.add_argument("--top_chain_var", dest="top_chain_var", default=0.005, type=float)
    args = parser.parse_args()

    executable = os.path.expanduser(os.path.join(args.dtm_path, "dtm", "main"))
    text_base = re.sub(r"-mult.dat", "", args.text_input)
    time_base = re.sub(r"-seq.dat", "", args.time_input)

    assert text_base == time_base

    arguments = [
        "--ntopics={}".format(args.topic_count),
        "--mode=fit",
        "--initialize_lda=true",
        "--corpus_prefix={}".format(time_base),
        "--outname=work/lda",
        "--top_chain_var={}".format(args.top_chain_var),
        "--alpha={}".format(args.alpha),
        "--lda_sequence_min_iter={}".format(args.min_lda_seq_iter),
        "--lda_sequence_max_iter={}".format(args.max_lda_seq_iter),
        "--lda_max_em_iter={}".format(args.max_lda_em_iter)
    ]

    tdir = tempfile.mkdtemp(prefix="dtm-")
    try:
        pid = subprocess.Popen(
            [executable] + arguments + ["--outname={}".format(tdir)],
        )
        pid.communicate()
        arcname = os.path.splitext(os.path.basename(args.output))[0]
        with tarfile.open(args.output, "w|gz") as tfd:
            tfd.add(tdir, arcname=arcname)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(tdir)

