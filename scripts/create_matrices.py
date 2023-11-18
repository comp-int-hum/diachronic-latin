import argparse
import math
import gzip
import json
import sys
import logging
import pickle
import pandas
import matplotlib
import numpy

logger = logging.getLogger("create_matrices")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--parse_stats", dest="parse_stats", help="Input file")
    parser.add_argument("--topic_annotations", dest="topic_annotations", help="Input file")    
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--window_size", dest="window_size", default=50, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    

    
    # text author title time window num
    docs = {}
    unique_times = set()
    unique_words = set()
    unique_topics = set()
    unique_authors = set()
    with gzip.open(args.topic_annotations, "rt") as ifd:
        for i, line in enumerate(ifd):
            #if i > 3000:
            #    break
            j = json.loads(line)
            title = j["title"]
            author = j["author"]
            year = j["time"]
            num = j["num"]            
            key = (title, author, year)
            docs[key] = docs.get(key, [])
            docs[key].append(j)
            unique_times.add(year)
            unique_authors.add(author)
            for w, t in j["text"]:
                if t != None:
                    unique_words.add(w)
                    unique_topics.add(t)

    sorted_times = list(sorted(unique_times))
    min_time = sorted_times[0]
    max_time = sorted_times[-1]

    time2window = {}
    cur_min_time = min_time
    cur_max_time = min_time
    unique_windows = set()
    for i in range(math.ceil((max_time - min_time + 1) / args.window_size)):
        cur_max_time += args.window_size
        j = 0
        while j < len(sorted_times) and sorted_times[j] < cur_max_time:
            time2window[sorted_times[j]] = i
            j += 1
            key = (cur_min_time, cur_max_time)
        sorted_times = sorted_times[j:]
        cur_min_time = cur_max_time

    nwins = len(set(time2window.values()))
    nwords = len(unique_words)
    ntopics = len(unique_topics)
    nauths = len(unique_authors)
    word2id = {w : i for i, w in enumerate(unique_words)}
    id2word = {i : w for w, i in word2id.items()}
    author2id = {a : i for i, a in enumerate(unique_authors)}
    id2author = {i : a for a, i in author2id.items()}

    
    logger.info(
        "Found %d windows, %d unique words, %d unique topics, and %d unique authors",
        nwins,
        nwords,
        ntopics,
        nauths
    )

    words_wins_topics = numpy.zeros(shape=(nwords, nwins, ntopics))
    auths_wins_topics = numpy.zeros(shape=(nauths, nwins, ntopics))
    #topics_wins_words = numpy.zeros(shape=(ntopics, nwins, nwords))
    #wins_avdiameters = numpy.zeros(shape=(nwins,))
    #wins_avsentencelengths = numpy.zeros(shape=(nwins,))
    #wins_avwordlengths = numpy.zeros(shape=(nwins,))
    #wins_avdepstddevs = numpy.zeros(shape=(nwins,))


    for (title, author, year), subdocs in docs.items():
        aid = author2id[author]
        win = time2window[year]
        
        for subdoc in subdocs:
            for word, topic in subdoc["text"]:
                if topic != None:
                    wid = word2id[word]
                    words_wins_topics[wid, win, topic] += 1
                    auths_wins_topics[aid, win, topic] += 1
    #                topics_wins_words[topic, win, wid] += 1
    #print(words_wins_topics == numpy.transpose(topics_wins_words, axes=(2, 1, 0)))
    #print(words_wins_topics.sum())
    
    window_sequences = {}
    
    # author title year diameters sentence_lengths word_lengths dependent_stddevs
    syntactic_measures = ["diameters", "sentence_lengths", "word_lengths", "dependent_stddevs"]
    with gzip.open(args.parse_stats, "rt") as ifd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            author = j["author"]
            title = j["title"]
            year = j["year"]
            #if year not in time2window:
            #    break
            window = time2window[year]
            window_sequences[window] = window_sequences.get(window, {})
            for meas in syntactic_measures:
                window_sequences[window][meas] = window_sequences[window].get(meas, [])
                for v in j[meas]:
                    window_sequences[window][meas].append(v)
    window_sequences = {w : {meas : sum(vals) / len(vals) for meas, vals in measures.items()} for w, measures in window_sequences.items()}



    
    matrices = {
        "start" : min_time,
        "window_size" : args.window_size,
        "id2author" : id2author,
        "id2word" : id2word,
        "wwt" : words_wins_topics,
        "awt" : auths_wins_topics,
        "syntax" : window_sequences
    }
            
    with gzip.open(args.output, "wb") as ofd:
        ofd.write(pickle.dumps(matrices))
