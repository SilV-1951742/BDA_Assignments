from functools import partial
import re
from html import unescape
import argparse
import matplotlib.pyplot as plt
from typing import NamedTuple, Final, List
import os
import time
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

BLACKLIST: Final = [".", "Chair's Message.", "Editor's Notes.", "Chairman's Column.", "Title, Contents.", "Chairman's Message.", "Editor's Remarks.", "Letter.", "Announcements.", "Meeting Announcements.", "Meeting Announcements and Calls for Papers.",  "-closeness.", "-diversity.", "From the Chairman.", "Notes from the Vice Chairperson.", "Availability Notices.", "Summary Tables.", "Technical Reports.", "Title", "Reports on Standards.", "Calls for Papers / Announcements.", "Correction and Announcements.", "Meeting and Course Announcements."]

class year_title_tuple(NamedTuple):
    """
    NamedTuple to hold title-year pairs
    """
    year: int
    title: str

class year_title_except(Exception):
    """Exception thrown if year/title can't be found."""
    pass

class year_title_generator:
    """
    Class to generate title_year tuples
    """
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

        if not title_r or not year_r:
            raise year_title_except()

        title_str = str(title_r.group(1))
        if title_str in BLACKLIST:
            raise year_title_except()
        
        return year_title_tuple(int(year_r.group(1)), title_str)

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
                    ty_tuple = self.create_year_title_tuple(entry)
                    if ty_tuple.title in BLACKLIST:
                        continue
                    yield ty_tuple
            except year_title_except:
                continue

    def __enter__(self):
        print(f"Opening {self.filename}")
        self.fp = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fp.close()
        return True

def generate_elbow_graph(pipeline, yt_col):
    """
    Function to generate elbow graphs of kmeans
    """
    print("Creating elbow graphs")
    #plt.figure()
    
    plot_index =  0
    fig, axes = plt.subplots(nrows=2, ncols=3)
    for y in range(1960, 2020, 10):
        features = pipeline.fit_transform( [year_title.title for year_title  in yt_col if
                                            (year_title.year >= y and year_title.year < (y + 15))])
        elbow_list = {}
        
        for k in range(2, 31):
            kmeans = cluster.MiniBatchKMeans(n_clusters=k, n_init=15, max_iter=200)
            kmeans.fit(features)
            #label =  kmeans.labels_
            #coeff = silhouette_score(features, label, metric='euclidean')
            elbow_list[k] = kmeans.inertia_
            #print(f"For n_clusters={k} in range {interval}, The Silhouette Coefficient is {coeff}")

        axes[plot_index // 3, plot_index % 3].set_title(f"{y} - {y+15}")
        axes[plot_index // 3, plot_index % 3].set_xlabel("Number of clusters")
        axes[plot_index // 3, plot_index % 3].set_ylabel("Inertia")
        axes[plot_index // 3, plot_index % 3].plot(list(elbow_list.keys()), list(elbow_list.values()))
        
        #plt.title(f"Elbow graph for time periode {y} - {y+15}")
        plot_index += 1
        
    fig.suptitle("Elbow graphs")
    plt.tight_layout()
    plt.savefig('elbow.pdf')
    plt.show()

def print_data(data, vectorizer):
    """
    Function to print cluster data
    """
    print("Printing data")

    X = vectorizer.fit_transform([year_title.title for year_title in data]).todense()
    
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    plt.scatter(data2D[:,0], data2D[:,1])
    plt.show()


def build_clusters(features, labels, centres):
    clusters = {}
    for i in range(len(labels)):
        if labels[i] in clusters:
            clusters[labels[i]].append(features[i])
        else:
            clusters[labels[i]] = [features[i]]
    return clusters


class StemmedHasher(HashingVectorizer):
    lemmatizer = WordNetLemmatizer()
    
    def build_analyzer(self):
        analyzer = super(StemmedHasher, self).build_analyzer()
        return lambda doc: (StemmedHasher.lemmatizer.lemmatize(w) for w in analyzer(doc))
    
    
def plot_tsne_pca(data, labels, title):
    """
    Function to plot both pca and tsne reduced data
    """
    pca = PCA(n_components=2).fit_transform(data.todense())
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=200).fit_transform(data)
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[:, 0], pca[:, 1], c=labels)
    ax[0].set_title(f"PCA {title}")
    
    ax[1].scatter(tsne[:, 0], tsne[:, 1], c=labels)
    ax[1].set_title(f"TSNE {title}")

    plt.show()

    
def main():
    args = arg_parser.parse_args()

    # Get title-year pairs
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


    # Setup the vectorizer
    my_stop_words = ENGLISH_STOP_WORDS.union(["application", "title", "contents", "editor", "chairman", "chair", "vice", "note", "conference"])
    hasher = StemmedHasher(n_features=10000, ngram_range=(1, 1), 
                           stop_words=my_stop_words, alternate_sign=False,
                           norm=None, binary=False)
    vect = make_pipeline(hasher, TfidfTransformer())
    
    # ----
    # ---- DBSCAN implementation, does not work well ----
    # ----
    # print("\r\nDBSCAN\r\n")
    # kmeans_cluster_size: List[int] = []
        # for y in range(1960, 2020, 10):
    #     try:
    #         print(f"DBSCAN in range of year {y} - {y+15}")
    #         features = vect.fit_transform([year_title.title for year_title  in year_title_collection if
    #                                        (year_title.year >= y and year_title.year < (y + 15))])

    #         db = cluster.DBSCAN(eps=0.1, min_samples=5)
    #         predict = db.fit_predict(features)
    #         labels = db.labels_

    #         no_clusters = len(np.unique(labels) )
    #         no_noise = np.sum(np.array(labels) == -1, axis=0)

            
    #         kmeans_cluster_size.append(no_clusters)

    #         print(f'Estimated no. of clusters: {no_clusters}')
    #         print(f'Estimated no. of noise points: {no_noise}')

    #         #tsne = TSNE(n_components=2, verbose=0, perplexity=5, n_iter=1000).fit_transform(features)
    #         pca = PCA(n_components=2).fit_transform(features.todense())
    #         plt.scatter(pca[:, 0], pca[:, 1], c=predict)

    #         plt.show()
    #     except ValueError:
    #         continue

    # print("\r\n---")
    # print("---")
    
    print("\r\nKMeans")

    print("Elbow criterion per time period")
    # Generate the elbow graphs, comment this out if you want to quickly run kmeans
    generate_elbow_graph(vect, year_title_collection)

    # Loop per time interval
    centres = [5, 8, 13, 14, 15, 16]
    cluster_index = 0
    # Amount of terms to print
    top_terms =  5
    for y in range(1960, 2020, 10):
        try:
            # Get the relevant entries
            features_pre_transform: List[str]  = [year_title.title for year_title  in year_title_collection if
                                                  (year_title.year >= y and year_title.year < (y + 15))]

            print(f"Clusters in range of year {y} - {y+15}\r\n{len(features_pre_transform)} titles in this range")
            # Vectorize the filtered entries
            features = vect.fit_transform(features_pre_transform)

            # Actual KMeans
            km = cluster.MiniBatchKMeans(n_clusters=centres[cluster_index], n_init=15, max_iter=200)
            y_kmeans = km.fit_predict(features)

            # Print the top titles per cluster
            final_clusters = build_clusters(features_pre_transform, y_kmeans, km.cluster_centers_)
            for i in range(centres[cluster_index]):
                fc_transformed = vect.transform(final_clusters[i])
                dist_centers = km.transform(fc_transformed)
                sorted = dist_centers[:, i].argsort()[:top_terms]
                
                print(f"Cluster {i}: ", end="")
                for index in sorted:
                    print(f"\"{final_clusters[i][index]}\" - ", end="")
                print()
                    
            cluster_index += 1

            # Plot kmeans result by reducing components
            # PCA is faster than TSNE
            #tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=2000).fit_transform(features)
            pca = PCA(n_components=2).fit_transform(features.todense())
            plt.scatter(pca[:, 0], pca[:, 1], c=y_kmeans)
            plt.tight_layout()
            plt.savefig(f'pca_{y}-{y + 15}.pdf')
            plt.show()
            print()
            print()
        except ValueError:
            continue

    
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
