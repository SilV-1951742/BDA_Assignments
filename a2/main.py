from functools import partial
import re
from html import unescape
import argparse
from sklearn import cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from typing import NamedTuple, Final
import os
import time
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

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

def tokenize(raw):
    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]

class StemmedTfidfVectorizer(TfidfVectorizer):
    lemmatizer = WordNetLemmatizer()
    
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (StemmedTfidfVectorizer.lemmatizer.lemmatize(w) for w in analyzer(doc))

    
def main():
    args = arg_parser.parse_args()
    year_title_collection = []
    with year_title_generator(args.dataset, args.chunksize) as titles:
        for title in titles.title_iterator():
            year_title_collection.append(title)

    print("Titles collected")
    print(f"Entries in filtered titles: {len(year_title_collection)}")

    my_stop_words = ENGLISH_STOP_WORDS.union(["application"])
    min_hasher = StemmedTfidfVectorizer(norm='l2', use_idf=True, stop_words=my_stop_words, ngram_range=(1, 1), tokenizer=tokenize)
    
    print("Hashed")

    for y in range(1960, 2020, 10):
        try:
            clusters: int = 10
            print(f"Clusters in range of year {y} - {y+15}")
            titles_hashed = min_hasher.fit_transform([year_title.title for year_title  in year_title_collection if
                                                      (year_title.year >= y and year_title.year < (y + 15))])
    
            km = cluster.KMeans(n_clusters=clusters)
            km.fit(titles_hashed)

            print(f"Top terms per cluster {y} - {y+15}:")
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            terms = min_hasher.get_feature_names()
            for i in range(clusters):
                top_five_words = [terms[ind] for ind in order_centroids[i, :5]]
                print("Cluster {}: {}".format(i, ' '.join(top_five_words)))
            print()
            print()
        except ValueError:
            continue

    # plt.scatter(titles_hashed[:,0],titles_hashed[:,1], c=km.labels_, cmap='rainbow')
    # plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')
    # plt.plot()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
