import argparse
#../dtm/dtm/main --ntopics=10 --mode=fit --initialize_lda=true --corpus_prefix=work/dtm_input --outname=work/lda --top_chain_var=0.005 --alpha=0.01 --lda_sequence_min_iter=6 --lda_sequence_max_iter=20 --lda_max_em_iter=10 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()
