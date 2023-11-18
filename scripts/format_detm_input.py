import argparse
import glob
import pickle
import os.path
import gzip
import json
import scipy.io


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest="data_path", help="Path to DETM-style input data")
    parser.add_argument("--train_output", dest="tr_output", help="Output file")
    parser.add_argument("--val_output", dest="va_output", help="Output file")
    parser.add_argument("--test_output", dest="ts_output", help="Output file")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, 'vocab.pkl'), 'rb') as ifd:
        vocab = pickle.load(ifd)

    for tp in ["tr", "va", "ts"]:
        with gzip.open(getattr(args, "{}_output".format(tp)), "wt") as ofd:
            token_file = os.path.join(args.data_path, 'bow_{}_tokens.mat'.format(tp))
            count_file = os.path.join(args.data_path, 'bow_{}_counts.mat'.format(tp))
            time_file = os.path.join(args.data_path, 'bow_tr_timestamps.mat'.format(tp))            
            tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
            counts = scipy.io.loadmat(count_file)['counts'].squeeze()
            times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
            for i in range(tokens.size):
                doc = []
                for j in range(tokens[i].size):
                    assert counts[i].shape[0] == 1
                    for k in range(counts[i][0, j]):
                        doc.append(vocab[tokens[i][0, j]])
                ofd.write(
                    json.dumps(
                        {
                            "inferred_year" : int(times[i]),
                            "author" : "ACL",
                            "title" : "ACL",
                            "content" : [doc]
                        }
                    ) + "\n"
                )

