import argparse
import math
import gzip
import json
import sys
import warnings
import pickle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
import numpy
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import entropy
import ruptures

warnings.simplefilter("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--num_top_words", dest="num_top_words", type=int, default=10)
    parser.add_argument("--step_size", dest="step_size", type=int, default=1)
    parser.add_argument("--word_image", dest="word_image", help="Output file")
    parser.add_argument("--word_detail_image", dest="word_detail_image", help="Output file")
    parser.add_argument("--temporal_image", dest="temporal_image", help="Output file")
    parser.add_argument("--author_image", dest="author_image", help="Output file")
    parser.add_argument("--author_histogram", dest="author_histogram", help="Output file")
    parser.add_argument("--latex", dest="latex", help="Output file")
    args = parser.parse_args()

    # load in the various precomputed counts/lookups/etc    
    label_size = 20
    tick_size = 20
    title_size = 25
    with gzip.open(args.input, "rb") as ifd:
        precomp = pickle.loads(ifd.read())        

    # unpack the precomputed info

    # the start year and years-per-window (integers)
    start = precomp["start"]
    window_size = precomp["window_size"]

    # dictionary lookups for author names and words
    id2author = precomp["id2author"]
    id2word = precomp["id2word"]
    word2id = {v : k for k, v in id2word.items()}

    # 3-d count matrices (this could have been one 4-d matrix, but since authors only occur in one window it would be inefficient)
    word_win_topic = precomp["wwt"]
    auth_win_topic = precomp["awt"]

    word_counts = word_win_topic.sum(1).sum(1)

    # top-level dictionary is window-to-metrics, a metric is a dictionary of name-to-value
    syntax = precomp["syntax"]

    # create some meaningful derived matrices

    # each window's distribution over topics
    win_topic = word_win_topic.sum(0)
    win_topic = (win_topic.T / win_topic.sum(1)).T

    # each author's distribution over topics
    auth_topic_dist = auth_win_topic.sum(1)
    auth_topic_dist = (auth_topic_dist.T / auth_topic_dist.sum(1)).T

    # each author's *preceding* window's distribution over topics
    auth_wins = auth_win_topic.sum(2).argmax(1)
    auth_prev_background_topic_dist = win_topic[auth_wins - 1]

    # each topic's top word ids
    topic_word = numpy.transpose(word_win_topic.sum(1), axes=(1, 0))
    topic_word_dist = (topic_word.T / topic_word.sum(1)).T
    topic_word_ids = numpy.flip(topic_word_dist.argsort(1)[:, -args.num_top_words:], 1)

    # each word's distribution over topics over time
    word_win = word_win_topic.sum(2)
    word_win_topic_dist = numpy.zeros(shape=word_win_topic.shape)
    for win in range(word_win.shape[1]):
        word_win_topic_dist[:, win, :] = (word_win_topic[:, win, :].T / word_win[:, win]).T
    word_win_topic_maxes = numpy.sort(word_win_topic_dist, axis=2)[:, :, -2:]

    words = []
    word_win_modality = (word_win_topic_maxes[:, :, -1] - word_win_topic_maxes[:, :, -2]) + (1 - (word_win_topic_maxes[:, :, -1] + word_win_topic_maxes[:, :, -2]))


    word_modality_changepoint_delta = numpy.zeros(shape=(word_win_modality.shape[0],))
    for i, mod in enumerate(word_win_modality):
        if numpy.isnan(mod).sum() < 3:
            modalities = numpy.array([v for v in mod if not numpy.isnan(v)])
            #ent = word_win_entropies[i]
            cps = ruptures.Dynp(model="l2", min_size=1, jump=1).fit_predict(modalities.T, 1)
            cp = cps[0]
            cpd = abs(modalities[:cp].mean() - modalities[cp:].mean())
            if word_counts[i] > 200 and cpd > 0.0:
                words.append((cpd, cp, modalities.mean(), modalities.std(), id2word[i], word_counts[i]))
                #print(id2word[i], word_counts[i])

    with open(args.latex, "wt") as latex_ofd, open(args.word_image, "wb") as word_image_ofd, open(args.temporal_image, "wb") as temporal_image_ofd, open(args.author_image, "wb") as author_image_ofd, open(args.word_detail_image, "wb") as word_detail_image_ofd, open(args.author_histogram, "wb") as author_histogram_ofd:

        width = 12
        height = 6


        # plot specific words, together
        #cps = list(sorted(set([cp for _, cp, _, _, _, _ in words])))
        #color_map = {k : colormaps["cividis"](v) for k, v in zip(cps, numpy.linspace(0.0, 1.0, len(cps)))}
        fig = Figure(figsize=(width, height))
        
        words_in_detail = [("figura", [(25, "rhetorical"), (32, "comparative"), (34, "procedural"), (44, "animate")]), ("effigies", [(0, "representational"), (44, "animate")])]
        colors = ["red", "blue"]
        colors = {
            44 : colormaps["Accent"](5),
            25 : colormaps["Accent"](1),
            32 : colormaps["Accent"](2),
            34 : colormaps["Accent"](0),
            0 : colormaps["Accent"](4),
        }
        lines = []
        line_labels = []
        for j, (word, tops) in enumerate(words_in_detail):
            ax = fig.add_subplot(2, 1, j + 1, frameon=False)
            ax.set_title("\"{}\"".format(word), fontsize=title_size)
            topic_win = word_win_topic[word2id[word]].T / word_win_topic[word2id[word]].T.sum(0)
            for tid, tname in tops:                
                topic = topic_win[tid]
                first = None
                for i, v in enumerate(topic):
                    if not numpy.isnan(v) and not first:
                        first = i
                topic[:first] = topic[first]
                line, = ax.plot(topic, color=colors[tid], linewidth=2)
                if j != 0 or tid != 44:
                    lines.append(line)
                    line_labels.append(tname)
            ax.set_yticks([0.0, 1.0])
            if j == 0:
                ax.set_xticks([])
            else:
                ax.set_xticks(
                    [i for i in range(topic.shape[0])],
                    [str(start + i * window_size) for i in range(topic.shape[0])],
                    #rotation=90,
                    #fontsize=tick_size
                )
                ax.set_xlabel("Year", fontsize=label_size)
        ax.legend(reversed(lines), reversed(line_labels), loc="center left", fontsize=15)
        fig.savefig(word_detail_image_ofd, bbox_inches="tight")
        #print(args.word_detail_image)
        #sys.exit()

        
        window_modality_counts = {}
        words_by_modality_shift = list(reversed(sorted(words)))
        for cpd, cp, mmean, mstd, w, c in words:
            year = start + window_size * cp
            window_modality_counts[year] = window_modality_counts.get(year, 0.0) + cpd

        # plot modality shift over time
        fig = Figure(figsize=(width, height))
        ax = fig.add_subplot(frameon=False)
        pairs = list(sorted(window_modality_counts.items()))

        ax.plot([x for x, _ in pairs], [y for _, y in pairs], linewidth=5)
        ax.set_xticks([x for x, _ in pairs], labels=[str(x) for x, _ in pairs], fontsize=tick_size)
        ax.set_xlabel("Year", fontsize=label_size)
        ax.set_ylabel("Amount of bimodal shift", fontsize=label_size)
        fig.savefig(temporal_image_ofd, bbox_inches="tight")


        latex_ofd.write("""\\begin{tabular}{l l l}\n""")
        latex_ofd.write("""\\hline\n""")
        latex_ofd.write("""Word & Changepoint & Delta \\\\\n""")
        latex_ofd.write("""\\hline\n""")
        for cpd, cp, _, _, w, c in words_by_modality_shift[:5]:
            year = start + window_size * cp
            latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(w, year, cpd))
        latex_ofd.write("""\\hline\n""")
        for cpd, cp, _, _, w, c in words_by_modality_shift[-5:]:
            year = start + window_size * cp
            latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(w, year, cpd))
        latex_ofd.write("""\\hline\n""")
        latex_ofd.write("""\\end{tabular}\n""")

        # plot where a few words show up
        cps = list(sorted(set([cp for _, cp, _, _, _, _ in words])))
        color_map = {k : colormaps["cividis"](v) for k, v in zip(cps, numpy.linspace(0.0, 1.0, len(cps)))}
        fig = Figure(figsize=(width, height))
        ax = fig.add_subplot(frameon=False)
        word_list = [w for _, _, _, _, w, _ in words_by_modality_shift]
        words_of_interest = [
            "forma",
            "species",
            "figura",
            "simulacrum",
            "statua",
            "effigies",
            "imago"
        ]
        ax.bar(
            [w for _, _, _, _, w, _ in words_by_modality_shift],
            [v for v, _, _, _, _, _ in words_by_modality_shift],
            color=[color_map[cp] for _, cp, _, _, _, _ in words_by_modality_shift],
            width=1.0,
            linewidth=0,
            antialiased=True
        )
        labels = []
        for w in words_of_interest:
           i = word_list.index(w)
           labels.append((w, i))
        ax.set_xticks(
            [i for _, i in labels],
            [w for w, _ in labels],
            rotation=90,
            fontsize=tick_size
        )
        ax.set_ylabel("Changepoint delta", fontsize=label_size)
        fig.savefig(word_image_ofd, bbox_inches="tight")




        jsds = jensenshannon(auth_prev_background_topic_dist, auth_topic_dist, axis=1)
        authors_by_novelty = []
        for i in jsds.argsort():
            authors_by_novelty.append((id2author[i], jsds[i], auth_wins[i]))
        authors_by_novelty = list(reversed(authors_by_novelty))


        latex_ofd.write("""\\begin{tabular}{l l}\n""")
        latex_ofd.write("""\\hline\n""")
        latex_ofd.write("""Author & JSD \\\\\n""")
        latex_ofd.write("""\\hline\n""")
        for name, val, cp in authors_by_novelty[:5]:
            year = start + window_size * cp
            latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(name, year, val))
        latex_ofd.write("""\\hline\n""")
        for name, val, cp in authors_by_novelty[-5:]:
            year = start + window_size * cp
            latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(name, year, val))
        latex_ofd.write("""\\hline\n""")
        latex_ofd.write("""\\end{tabular}\n""")


        # plot where Lucretius/Varo/Cicero show up
        cps = list(sorted(set([cp for _, _, cp in authors_by_novelty])))
        color_map = {k : colormaps["cividis"](v) for k, v in zip(cps, numpy.linspace(0.0, 1.0, len(cps)))}
        fig = Figure(figsize=(width, height))
        ax = fig.add_subplot(frameon=False)
        authors_of_interest = [
            ("M. Terentius Varro", "Varro"),
            ("Lucretius", "Lucretius"),
            ("M. Tullius Cicero", "Cicero")
        ]
        labels = []
        author_list = [n for n, _, _ in authors_by_novelty]
        for name, label in authors_of_interest:
            i = author_list.index(name)
            labels.append((label, i))

        #ax = fig.add_subplot(gs[2, :], frameon=False)
        ax.bar(
            [str(n) for n, _, _ in authors_by_novelty],
            [v for _, v, _ in authors_by_novelty],
            color=[color_map[cp] for _, _, cp in authors_by_novelty],
            width=1.0,
            linewidth=0                
        )
        ax.set_xticks([i for _, i in labels], [l for l, _ in labels], rotation=90, fontsize=tick_size)
        ax.set_ylabel("Previous topic JSD", fontsize=label_size)
        fig.savefig(author_image_ofd, bbox_inches="tight")


        # topic evolutions
        topic_win_word = numpy.transpose(word_win_topic, axes=(2, 1, 0))
        indices = numpy.arange(topic_win_word.shape[1], step=args.step_size)

        for tid, topic in enumerate(topic_win_word):
            tt = (topic.T / topic.sum(1)).T
            top_words = numpy.flip(topic.argsort(1), 1)

            topic_states = []
            for j in indices:
                word_ids = top_words[j][:args.num_top_words]
                #topic_states.append(["{}:{:.03}".format(id2word[wid], tt[j][wid]) for wid in word_ids])
                topic_states.append(["{}".format(id2word[wid]) for wid in word_ids])

            latex_ofd.write(
                """\\topicevolution{%s}{0}{0}{{%s}}\n""" % (
                    tid,
                    ",".join(["{%s}" % (",".join(state)) for state in topic_states])
                ) + "\n"
            )
            
        abn = [(n, s, n in xians) for n, s, _ in authors_by_novelty]

        cps = list(sorted(set([cp for _, _, cp in authors_by_novelty])))
        color_map = {k : colormaps["cividis"](v) for k, v in zip(cps, numpy.linspace(0.0, 1.0, len(cps)))}
        fig = Figure(figsize=(width, height))
        ax = fig.add_subplot(frameon=False)
        ax.hist(
            [
                numpy.array([j for _, j, c in abn if c]),
                numpy.array([j for _, j, c in abn if not c])                
            ],
            color=["red", "blue"],
            label=["Christian", "Pagan"],
            bins=10
        )
        ax.legend(fontsize=15)
        ax.set_ylabel("Author count", fontsize=label_size)
        ax.set_xlabel("Novelty", fontsize=label_size)
        fig.savefig(author_histogram_ofd, bbox_inches="tight")
