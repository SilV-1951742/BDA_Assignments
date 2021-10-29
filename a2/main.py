from functools import partial
import re
from html import unescape
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import NamedTuple, Final, List
import os
import time
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import pairwise
from sklearn import cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import pandas as pd
import pickle


arg_parser = argparse.ArgumentParser(description="BDA opdr1.")

arg_parser.add_argument("dataset", type=str, help="Dataset source file.")

arg_parser.add_argument("chunksize", type=int, help="Chunksize in megabytes.")

arg_parser.add_argument(
    "--make_testfile",
    action="store_true",
    dest="testfile",
    help="Create a test dataset of 10000 entries.",
)

DATA_DB_BOOL: Final = False

DATA_MINING_KEYS: Final = ["journals/sigkdd", "conf/pkdd", "conf/icdm", "conf/sdm"]
DATA_MINING_REGEX: Final =  re.compile(r"(journals\/sigkdd|conf\/pkdd|conf\/icdm|conf\/sdm)")
DB_KEYS: Final = ["journals/sigmod", "journals/vldb", "conf/edbt", "conf/icde"]
DB_REGEX: Final =  re.compile(r"(journals\/sigmod|journals\/vldb|conf\/edbt|conf\/icde)")

class year_title_tuple(NamedTuple):
    year: int
    title: str

class year_title_except(Exception):
    """Exception thrown if year/title can't be found."""
    pass

class year_title_generator:
    def __init__(self, filename: str, chunk_size: int):
        self.filename = filename
        print(f"Filesize: {os.path.getsize(filename)//(1024*1024)} MB")
        self.chunk_size = chunk_size
        self.entry_regex = re.compile(
            r"(<article|<book|<phdthesis|<www|<incollection|<proceedings|<inproceedings)([\s\S]*?)(<\/article>|<\/book>|<\/phdthesis>|<\/www>|<\/incollection>|<\/proceedings>|<\/inproceedings>)"
)
        self.title_regex = re.compile(r"<title.*>(.*)<\/title>")
        self.year_regex = re.compile(r"<year.*>(.*)<\/year>")

    def entry_string_generator(self):
        previous = ""
        start_prev = 0
        file_r = partial(self.fp.read, self.chunk_size)
        for data in iter(file_r, ""):
            data = previous + data
            for result in re.finditer(self.entry_regex, data):
                start_prev = result.end()
                yield result.group(0)
            previous = data[start_prev:]

    def create_year_title_tuple(self, xml_element) -> year_title_tuple:
        title_r = re.search(self.title_regex, xml_element)
        year_r = re.search(self.year_regex, xml_element)

        if not title_r  or not year_r:
            raise year_title_except()
        
        return year_title_tuple(int(year_r.group(1)), str(title_r.group(1)))

    def check_key(self, xml_element) -> bool:
        if DATA_DB_BOOL == True:
            regex_match = re.search(DATA_MINING_REGEX, xml_element)
            if regex_match:
                return True
            return False
        else:
            regex_match = re.search(DB_REGEX, xml_element)
            if regex_match:
                return True
            return False

    def clean_entry(self, entry_string: str) -> str:
        return unescape(entry_string).replace("&", "&#38;")
    
    def title_iterator(self):
        print(DATA_MINING_KEYS)
        print(DB_KEYS)
        iter = 0
        start_time = time.time()
        for entry in self.entry_string_generator():
            iter += 1
            if iter%10000 == 0: 
                print(f"{iter} entries processed in {time.time() - start_time} seconds.")
            try:
                if(self.check_key(entry)):
                    yield self.create_year_title_tuple(entry)
            except year_title_except:
                continue

    def __enter__(self):
        print(f"Opening {self.filename}")
        self.fp = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()
        return True

def generate_elbow_graph(pipeline, year_title_list):
    print("Creating elbow graph")
    plt.figure()
    # for y in range(1960, 2020, 10):
        # try:
            # titles_hashed = hasher.fit_transform([year_title.title for year_title  in year_title_list if
            #                                   (year_title.year >= y and year_title.year < (y + 15))])
    titles_hashed = pipeline.fit_transform([year_title.title for year_title in year_title_list])
    elbow_list = {}
    legend_list = []
    for k in range(1, 20):
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(titles_hashed)
        elbow_list[k] = kmeans.inertia_
                
    plt.plot(list(elbow_list.keys()), list(elbow_list.values()))
    
    #legend_list.append(f"{y} - {y+15}")
        #except ValueError:
         #   continue
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title(f"Elbow graph")
    plt.show()


def tokenize(raw):
    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]


def build_clusters(features, labels):
    clusters = {}
    for i in range(len(labels)):
        if labels[i] in clusters:
            clusters[labels[i]].append(features[i])
        else:
            clusters[labels[i]] = [features[i]]
    return clusters


class StemmedTfidfVectorizer(TfidfVectorizer):
    lemmatizer = WordNetLemmatizer()
    
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (StemmedTfidfVectorizer.lemmatizer.lemmatize(w) for w in analyzer(doc))
    
def plot_tsne_pca(data, labels, title):
    max_label = max(labels)
    #max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data.todense())
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000).fit_transform(data)
    
    
    #idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    #label_subset = labels[data.shape[0]]
    # label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[:, 0], pca[:, 1], c=labels)
    ax[0].set_title(f"PCA {title}")
    
    ax[1].scatter(tsne[:, 0], tsne[:, 1], c=labels)
    ax[1].set_title(f"TSNE {title}")

    plt.show()

    
def main():
    args = arg_parser.parse_args()
    year_title_collection = []
    try:
        year_title_collection = pickle.load(open("year_title.pkl", "rb"))
    except (OSError, IOError) as e:
        with year_title_generator(args.dataset, args.chunksize) as titles:
            for title in titles.title_iterator():
                year_title_collection.append(title)
        pickle.dump(year_title_collection, open("year_title.pkl", "wb"))
    
    print("Titles collected")
    print(f"Entries in filtered titles: {len(year_title_collection)}")

    my_stop_words = ENGLISH_STOP_WORDS.union(["application"])
    vect = StemmedTfidfVectorizer(norm='l2', use_idf=True, stop_words=my_stop_words, ngram_range=(1, 1), tokenizer=tokenize)
    
    # print("Printing data")

    # X = vect.fit_transform([year_title.title for year_title in year_title_collection]).todense()
    
    # pca = PCA(n_components=2).fit(X)
    # data2D = pca.transform(X)
    # plt.scatter(data2D[:,0], data2D[:,1])
    # plt.show()

    # generate_elbow_graph(vect, year_title_collection)

    print("\r\nDBSCAN")
    kmeans_cluster_size: List[int] = []
    cluster_index = 0
    for y in range(1960, 2020, 10):
        try:
            print(f"Clusters in range of year {y} - {y+15}")
            features = vect.fit_transform([year_title.title for year_title  in year_title_collection if
                                                      (year_title.year >= y and year_title.year < (y + 15))])

            db = cluster.DBSCAN(eps=0.3, min_samples=10)
            db.fit(features)
            labels = db.labels_

            no_clusters = len(np.unique(labels) )
            no_noise = np.sum(np.array(labels) == -1, axis=0)
            kmeans_cluster_size.append(no_clusters)

            print(f'Estimated no. of clusters: {no_clusters}')
            print(f'Estimated no. of noise points: {no_noise}')
        except ValueError:
            continue

    print("---")
    print("---")
    print("\r\nKMeans")

    for y in range(1960, 2020, 10):
        try:
            top_terms =  10
           
            print(f"Clusters in range of year {y} - {y+15}")

            features_pre_transform: List[str]  = [year_title.title for year_title  in year_title_collection if
                                           (year_title.year >= y and year_title.year < (y + 15))]
            print(f"# features pre transfrom {features_pre_transform}")
            features = vect.fit_transform(features_pre_transform)

            km = cluster.KMeans(n_clusters=kmeans_cluster_size[cluster_index])
            y_kmeans = km.fit_predict(features)

            print(f"Top terms per cluster {y} - {y+15}:")
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            
            terms = vect.get_feature_names_out()

            final_clusters = build_clusters(features_pre_transform, y_kmeans)
            for i in range(clusters):
                sample = final_clusters[i][:3]
                # top_five_words = [terms[ind] for ind in order_centroids[i, :top_terms]]
                print(f"Cluster {i}: {sample}")

            plot_tsne_pca(features, y_kmeans, f'Clusters from {y} - {y + 15}')
            cluster_index += 1
            # # reduce the features to 2D
            # pca = PCA(n_components=2, random_state=random_state)
            # reduced_features = pca.fit_transform(features.toarray())

            # # reduce the cluster centers to 2D
            # reduced_cluster_centers = pca.transform(km.cluster_centers_)

            # plt.title(f'Clusters from {y} - {y + 15}')
            
            # plt.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(features))
            # plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
            
            # plt.show()
            print()
            print()
        except ValueError:
            continue

    
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
