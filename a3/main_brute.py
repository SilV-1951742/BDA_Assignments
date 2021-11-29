from typing import NamedTuple, Final, List, OrderedDict
from lxml import etree
from itertools import islice, chain, combinations
import argparse
import os
import traceback
import bleach
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import random
import hashlib
import matplotlib.pyplot as plt
from collections import OrderedDict

from nltk.util import pr

SHINGLE_SIZE: Final = 5
SAMPLES: Final = 1000

arg_parser = argparse.ArgumentParser(description="BDA assignment 2.")

arg_parser.add_argument("dataset", type=str, help="Dataset source file.")

arg_parser.add_argument("chunksize", type=int, help="Chunksize of batch parser.")

arg_parser.add_argument(
    "--make_testfile",
    action="store_true",
    dest="testfile",
    help="Create a test dataset of 10000 entries.",
)

class comment_tuple(NamedTuple):
    id: int
    #owner_id: int
    post_type: int
    score: int
    text: str

    
class shingle_set(NamedTuple):
    id: int
    shingles: frozenset[tuple]

class similarity(NamedTuple):
    id_set1: int
    id_set2: int
    similarity: float

    
class post_generator:
    """
    Class that return a generator to create shingles per post.
    """
    def __init__(self, filename: str, chunksize: int = 100):
        self.filename = filename
        self.chunksize = chunksize

    def __enter__(self):
        print(f"Opening file {self.filename} of size {os.path.getsize(self.filename)//(1024*1024)} MB.")
        #self.fp = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #self.fp.close()
        return True

    def parse(self) -> comment_tuple:
        try:
            context = etree.iterparse(self.filename, events=("end", ))
            for _,  elem in context:
                if elem.tag == "row":
                    assert elem.text is None, "The row wasn't empty"
                    yield comment_tuple(int(elem.attrib["Id"]),
                                        #int(elem.attrib["OwnerUserId"]),
                                        int(elem.attrib["PostTypeId"]),
                                        int(elem.attrib["Score"]),
                                        bleach.clean(elem.attrib["Body"])) 

                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                    
            del context
        except Exception:
            traceback.print_exc()
            
            
    def batch(self):
        parse_iter = iter(self.parse())

        for first in parse_iter:
            for p in chain([first], islice(parse_iter, self.chunksize)):
                yield p


class shingler:
    """
    Class that contain a tokenizer and stopwords to make shingling easier.
    """
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return True

    def tokenize(self, text: str) -> List[str]:
        text = text.translate(str.maketrans('', '', string.punctuation))
        # text_nop = text.split()
        text_nop = word_tokenize(text)
        filtered_words = []

        for word in text_nop:
            if word not in self.stopwords:
                filtered_words.append(word.lower())
        
        return filtered_words

    def create_shingle(self, input_comment: comment_tuple, shingle_size: int) -> frozenset[tuple]:
        tokens = self.tokenize(input_comment.text)
        comment_length = len(tokens)
        shingles =  frozenset(tuple(tokens[i:(i + shingle_size)]) for i in range(comment_length - shingle_size + 1))
       
        return shingle_set(input_comment.id,
                           shingles)

    
def hash_shingle_set(shingles: shingle_set) -> shingle_set:
    hashed_shingle_list = []
    #print(shingles)
    
    for shingle in shingles.shingles:
        m = hashlib.sha256()
        for elem in shingle:
            m.update(elem.encode())
        hashed_shingle_list.append(int.from_bytes(m.digest()[:4], 'little'))

    #print(hashed_shingle_list)
    return shingle_set(shingles.id,
                       frozenset(hashed_shingle_list))

def calculate_jaccard_similarity(set1: shingle_set, set2: shingle_set) -> float:
    intersection = frozenset.intersection(set1.shingles, set2.shingles)
    union = frozenset.union(set1.shingles, set2.shingles)

    if(len(union) == 0):
        return 0
    
    return len(intersection) / len(union)

        
def main():
    print("Starting BDA assignment 3.")
    
    args = arg_parser.parse_args()
    comment_amount = 0
    comment_list = []
    rnd_comments = []
    shingles = []
    hashed_shingles = []
    frequencies = []

    # ------- Test with small data
    # shingles.append(shingle_set(0,
    #                             frozenset([tuple(["first", "shingle"])])))
    # shingles.append(shingle_set(1,
    #                             frozenset([tuple(["second", "shingle"])])))

    # shingles.append(shingle_set(2,
    #                             frozenset([tuple(["first", "shingle"]),
    #                                        tuple(["second", "shingle"]),
    #                                        tuple(["random", "word"])])))
    # shingles.append(shingle_set(3,
    #                             frozenset([tuple(["second", "shingle"])])))

    # for shingle in shingles:
    #     hashed_shingles.append(hash_shingle_set(shingle))

    # for comb in combinations(hashed_shingles, 2):
    #     print(comb)
    #     frequencies.append(similarity(comb[0].id,
    #                                   comb[1].id, 
    #                                   calculate_jaccard_similarity(comb[0], comb[1])))
    # ------- End of test

    
    # zero_freq = 0
    # for freq in frequencies:
    #     if freq.similarity > 0.0:
    #         print(freq)
    #         for comment in rnd_comments:
    #             if comment.id == freq.id_set1:
    #                 print(f"Comment: {comment}")
    #             if comment.id == freq.id_set2:
    #                 print(f"Comment: {comment}")
    #         print()
    #         for shingle in shingles:
    #             if shingle.id == freq.id_set1:
    #                 print(f"Shingle: {shingle}")
    #             if shingle.id == freq.id_set2:
    #                 print(f"Shingle: {shingle}")
    #         print()
    #         for shingle in hashed_shingles:
    #             if shingle.id == freq.id_set1:
    #                 print(f"Shingle: {shingle}")
    #             if shingle.id == freq.id_set2:
    #                 print(f"Shingle: {shingle}")
    #         print()
    #         print()
    #         print()
    #         print()
    #     else:
    #         zero_freq += 1
    # print(zero_freq)
    
    # Get posts from xml file
    with post_generator(args.dataset, args.chunksize) as posts:
        for post in posts.batch():
            comment_list.append(post)

            
    comment_amount = len(comment_list)
    print(f"Found {comment_amount} comments.")

    
    # Get random samples
    try:
        random_indices = random.sample(range(comment_amount), SAMPLES)

        for index in random_indices:
            rnd_comments.append(comment_list[index])
    except ValueError:
        print("Can't get random sample, count larger than population.")
        exit()

        
    # Create shingles
    with shingler() as shingle_creator:
        for comment in rnd_comments:
            shingles.append(shingle_creator.create_shingle(comment, SHINGLE_SIZE))

    print("Finished shingling")
            
    # Hash shingles
    for shingle in shingles:
        hashed_shingles.append(hash_shingle_set(shingle))
    print("Shingles hashed")

    # Calculate similarities
    for comb in combinations(hashed_shingles, 2):
        frequencies.append(similarity(comb[0].id,
                                      comb[1].id, 
                                      calculate_jaccard_similarity(comb[0], comb[1])))
    histogram_data = dict()

    zero_freq = 0
    for freq in frequencies:
        if freq.similarity > 0.0:
            current_sim = round(freq.similarity * 100, 1)

            if current_sim in histogram_data:
                histogram_data[current_sim] += 1
            else:
                histogram_data[current_sim] = 1

            # print(freq)
            # for comment in rnd_comments:
            #     if comment.id == freq.id_set1:
            #         print(f"Comment: {comment}")
            #     if comment.id == freq.id_set2:
            #         print(f"Comment: {comment}")
            # print()
            # for shingle in shingles:
            #     if shingle.id == freq.id_set1:
            #         print(f"Shingle: {shingle}")
            #     if shingle.id == freq.id_set2:
            #         print(f"Shingle: {shingle}")
            # print()
            # for shingle in hashed_shingles:
            #     if shingle.id == freq.id_set1:
            #         print(f"Shingle: {shingle}")
            #     if shingle.id == freq.id_set2:
            #         print(f"Shingle: {shingle}")
            # print()
            # print()
            # print()
            # print()
        else:
            zero_freq += 1

    print("histogram data")
    print(histogram_data)
    print("Amount of combinations with zero similarity: " + str(zero_freq))

    sorted_data = OrderedDict(sorted(histogram_data.items(), key=lambda t: t[0]))

    plt.figure()
    plt.bar([str(i) for i in sorted_data.keys()], sorted_data.values(), color='g')

    plt.show()

    
            
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
