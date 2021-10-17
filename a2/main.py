from functools import partial
import re
from html import unescape
import argparse
import itertools
from typing import Final
from sklearn import cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt


arg_parser = argparse.ArgumentParser(description="BDA opdr1.")

arg_parser.add_argument("dataset", type=str, help="Dataset source file.")

arg_parser.add_argument("chunksize", type=int, help="Chunksize in megabytes.")

arg_parser.add_argument(
    "--make_testfile",
    action="store_true",
    dest="testfile",
    help="Create a test dataset of 10000 entries.",
)

class title_generator:
    def __init__(self, filename: str, chunk_size: int):
        self.filename = filename
        self.chunk_size = chunk_size
        self.entry_regex = re.compile(
                r"<(article|book|phdthesis|www|incollection|proceedings|inproceedings)[\s\S]*?<(\/article|\/book|\/phdthesis|\/www|\/incollection|\/proceedings|\/inproceedings)>"
)
        self.title_regex = re.compile(r"<title>([\s\S]*?)<\/title>")

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

    def create_title_string(self, xml_string: str) -> str:
        return self.title_regex.search(xml_string).group(1)

    def title_iterator(self):
        for entry in self.entry_string_generator():
            yield self.create_title_string(unescape(entry))

    def __enter__(self):
        self.fp = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()
        return True

stemmer = SnowballStemmer('english')

def stemmer_tokenizer(input_string: str):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", input_string).lower().split()
    return [stemmer.stem(word) for word in words]

    
def main():
    args = arg_parser.parse_args()

    title_collection = []
    with title_generator(args.dataset, args.chunksize) as titles:
        for title in titles.title_iterator():
            title_collection.append(title)

    print("Titles collected")
    
    min_hasher = TfidfVectorizer(norm='l1', use_idf=False, stop_words='english')
    titles_hashed = min_hasher.fit_transform(title_collection)

    print("Hashed")
    
    km = cluster.KMeans(n_clusters=20)
    km.fit(titles_hashed)

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = min_hasher.get_feature_names()
    for i in range(20):
        top_five_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_five_words)))

    # plt.scatter(titles_hashed[:,0],titles_hashed[:,1], c=km.labels_, cmap='rainbow')
    # plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')
    # plt.plot()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
