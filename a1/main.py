from functools import partial
import re
import xml.etree.ElementTree as ET
from html import unescape
import argparse
import os
import pickle
import itertools


entry_regex = re.compile(
    r"<(article|book|phdthesis|www|incollection|proceedings|inproceedings)[\s\S]*?<(\/article|\/book|\/phdthesis|\/www|\/incollection|\/proceedings|\/inproceedings)>"
)

author_regex = re.compile(r"<author>([\s\S]*?)<\/author>")
title_regex = re.compile(r"<title>([\s\S]*?)<\/title>")

arg_parser = argparse.ArgumentParser(description="BDA opdr1.")

arg_parser.add_argument("dataset", type=str, help="Dataset source file.")

arg_parser.add_argument("chunksize", type=int, help="Chunksize in megabytes.")

arg_parser.add_argument(
    "--make_testfile",
    action="store_true",
    dest="testfile",
    help="Create a test dataset of 10000 entries.",
)


def entry_string(filename: str, chunk_size: int):
    """
    This function takes a filename and chunk size an return a generator
    that yields entries in the database.
    """
    with open(filename, "r") as f:
        previous = ""
        start_prev = 0
        file_r = partial(f.read, chunk_size)

        for data in iter(file_r, ""):
            data = previous + data
            for result in re.finditer(entry_regex, data):
                start_prev = result.end()
                yield result.group(0)
            previous = data[start_prev:]


def entry_tree(xml_string: str):
    return ET.fromstring(xml_string.replace("&", ""))


def create_author_set(xml_string: str) -> frozenset:
    """
    This function takes an entry and returns the author set.
    """
    author_set = set()
    for author in re.finditer(author_regex, xml_string):
        author_set.add(author.group(1))
    return frozenset(author_set)


def create_testfile(dataset: str, chunksize: int):
    """
    This function creates a testfile with 10000 entries to test
    a maximal frequent itemset algorithm.
    """
    try:
        os.remove("testfile.xml")
    except OSError:
        pass

    gen_entry_string = entry_string(dataset, chunksize * 1024 * 1024)
    author_set_list = []

    with open("testfile.xml", "a") as f:
        for _ in range(10000):
            tmp_entry = unescape(next(gen_entry_string))
            author_set_list.append(create_author_set(tmp_entry))
            f.write(tmp_entry)


def count_singletons(set_list):
    """
    This function takes a list of author sets and creates a picklefile
    with a dictionary with the counted singleton authors.
    """
    singleton_dict = dict()
    for sets in set_list:
        for elem in sets:
            if elem in singleton_dict:
                singleton_dict[elem] += 1
            else:
                singleton_dict[elem] = 1

    with open("singletons.pkl", "wb") as pkl_file:
        pickle.dump(singleton_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    return singleton_dict


def gen_candidate_k_tuple(singletons, min_support, k, previous_iteration = dict()):
    """
    Generator to calculate k sized tuples.
    """
    for comb in itertools.combinations(singletons, k):
        for elem in comb:
            if singletons[elem] < min_support:
                continue
            if len(previous_iteration) > 0:
                print("Do something")
        yield frozenset(comb)


def gen_counted_pairs(singletons, min_support, set_list):
    """
    Function that creates a dictionary of counted pairs.
    """
    pairs = []
    pair_dict = dict()

    if os.path.exists("pairs.pkl"):
        print("Opening pairs pickle file.")
        with open("pairs.pkl", "rb") as pkl_file:
            pair_dict = pickle.load(pkl_file)
    else:
        pair_generator = gen_candidate_k_tuple(singletons, min_support, 2)
        for comb in pair_generator:
            if comb not in pairs:
                pairs.append(comb)

        print("Generated pair candidates")

        for pair in pairs:
            for elem in set_list:
                if pair.issubset(elem):
                    print(f"Found matching pair {pair} in {elem}")
                    if pair in pair_dict:
                        pair_dict[pair] += 1
                    else:
                        pair_dict[pair] = 1

        with open("pairs.pkl", "wb") as pkl_file:
            pickle.dump(pair_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    return pair_dict

def gen_counted_tuples(singletons, pair_dict, min_support, set_list):
    k = 3

    candidate_tuples = []
    candidate_dict = dict()

    current_tuples = pair_dict.items()
    candidate_dict = pair_dict
    
    while len(current_tuples) != 0:
        tuple_generator = gen_candidate_k_tuple(singletons, min_support, k)
        for comb in tuple_generator:
            if comb not in candidate_tuples

# def apriori(singletons, min_support):
#     k = 2


def main():
    args = arg_parser.parse_args()

    if args.testfile:
        create_testfile(args.dataset, args.chunksize)
        exit()

    author_set_list = []
    freq_singletons = dict()
    support = 4

    try:
        gen_entry_string = entry_string(args.dataset, args.chunksize * 1024 * 1024)
        for entry in gen_entry_string:
            tmp_author_set = create_author_set(unescape(entry))
            if len(tmp_author_set) != 0:
                author_set_list.append(tmp_author_set)
    except FileNotFoundError:
        print(f"Could not find file {args.dataset}")

    try:
        print("Opening frequent singletons.")
        with open("freq_singletons.pkl", "rb") as pkl_file:
            freq_singletons = pickle.load(pkl_file)
    except FileNotFoundError:
        print("Can't open frequent singletons file")
        freq_singletons = dict(
            filter(
                lambda elem: elem[1] >= support,
                count_singletons(author_set_list).items(),
            )
        )
        with open("freq_singletons.pkl", "wb") as pkl_file:
            pickle.dump(freq_singletons, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Amount of freq singletons: {len(freq_singletons)}")

    freq_pairs = dict(
        filter(
            lambda elem: elem[1] >= support,
            gen_counted_pairs(freq_singletons, 4, author_set_list).items(),
        )
    )

    print(freq_pairs)

    # with open("singletons.pkl", "rb") as pkl_file:
    #     singleton_dict = pickle.load(pkl_file)
    #     freq_items = dict(
    #         filter(lambda elem: elem[1] >= support, singleton_dict.items())
    #     )
    #     print(freq_items)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
