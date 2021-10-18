from functools import partial
import re
from html import unescape
import argparse
from sklearn import cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from typing import NamedTuple


arg_parser = argparse.ArgumentParser(description="BDA opdr1.")

arg_parser.add_argument("dataset", type=str, help="Dataset source file.")

arg_parser.add_argument("chunksize", type=int, help="Chunksize in megabytes.")

arg_parser.add_argument(
    "--make_testfile",
    action="store_true",
    dest="testfile",
    help="Create a test dataset of 10000 entries.",
)

class year_title_tuple(NamedTuple):
    year: int
    title: str

class year_title_generator:
    def __init__(self, filename: str, chunk_size: int):
        self.filename = filename
        self.chunk_size = chunk_size
        self.entry_regex = re.compile(
                r"<(article|book|phdthesis|www|incollection|proceedings|inproceedings)[\s\S]*?<(\/article|\/book|\/phdthesis|\/www|\/incollection|\/proceedings|\/inproceedings)>"
)
        self.title_regex = re.compile(r"<title>([\s\S]*?)<\/title>")
        self.year_regex = re.compile(r"<year>([\s\S]*?)<\/year>")

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

    def create_year_title_tuple(self, xml_string: str) -> year_title_tuple:
        year: int = int(self.year_regex.search(xml_string).group(1))
        title: str = self.title_regex.search(xml_string).group(1)
        return year_title_tuple(year, title)

    def title_iterator(self):
        for entry in self.entry_string_generator():
            yield self.create_year_title_tuple(unescape(entry))

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

    year_title_collection = []
    with year_title_generator(args.dataset, args.chunksize) as titles:
        for title in titles.title_iterator():
            # print(title)
            year_title_collection.append(title)

    print("Titles collected")
    
    min_hasher = TfidfVectorizer(norm='l2', use_idf=True, stop_words='english', ngram_range=(1, 1))
    
    print("Hashed")

    for y in range(1960, 2020, 10):
        clusters: int = 10
        print(f"Clusters in range of year {y} - {y+10}")
        titles_hashed = min_hasher.fit_transform([year_title.title for year_title  in year_title_collection if
                                                  (year_title.year >= y and year_title.year < (y + 10))])
    
        km = cluster.KMeans(n_clusters=clusters)
        km.fit(titles_hashed)

        print(f"Top terms per cluster {y} - {y+10}:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = min_hasher.get_feature_names()
        for i in range(clusters):
            top_five_words = [terms[ind] for ind in order_centroids[i, :5]]
            print("Cluster {}: {}".format(i, ' '.join(top_five_words)))
        print()
        print()

    # plt.scatter(titles_hashed[:,0],titles_hashed[:,1], c=km.labels_, cmap='rainbow')
    # plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')
    # plt.plot()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
