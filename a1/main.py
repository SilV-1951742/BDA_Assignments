from functools import partial
import re
from html import unescape
import argparse
import os
import pickle
import itertools
import hashlib
from typing import Final
from collections import defaultdict

HASH_BUCKETS: Final = 10000

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


def hash_tuple(tuple_object):
    tuple_bytes = bytearray(str(tuple_object), "utf-8")
    return int(hashlib.sha256(tuple_bytes).hexdigest(), 16) % HASH_BUCKETS


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
        for _ in range(70000):
            tmp_entry = unescape(next(gen_entry_string))
            author_set_list.append(create_author_set(tmp_entry))
            f.write(tmp_entry)


def count_singletons(set_list):
    """
    This function takes a list of author sets and creates a picklefile
    with a dictionary with the counted singleton authors.
    """
    singleton_dict = dict()
    pair_hash_dict = defaultdict()

    for sets in set_list:
        for elem in sets:
            if elem not in singleton_dict:
                singleton_dict[elem] = 1
            else:
                singleton_dict[elem] += 1

        for comb in itertools.combinations(sets, 2):
            pair_hash = hash_tuple(comb)
            if pair_hash not in pair_hash_dict:
                pair_hash_dict[pair_hash] = 1
            else:
                pair_hash_dict[hash_tuple(comb)] += 1

    return singleton_dict, pair_hash_dict


def gen_candidate_k_tuple(previous_iteration, k: int, min_support, pair_hash=dict()):
    """
    Generator to calculate k sized tuples.
    """
    for pairs in itertools.combinations(previous_iteration, 2):
        if k > 2:
            for comb in itertools.combinations(pairs[0].union(pairs[1]), k):
                yield frozenset(comb)
        else:
            try:
                if pair_hash[hash_tuple(pairs)] < min_support:
                    continue
                else:
                    yield frozenset(pairs)
            except KeyError:
                continue


def gen_counted_pairs(singletons, set_list, min_support, pair_hash):
    """
    Function that creates a dictionary of counted pairs.
    """
    pairs = []
    pair_dict = defaultdict()

    pair_generator = gen_candidate_k_tuple(singletons, 2, min_support, pair_hash)
    for comb in pair_generator:
        if comb not in pairs:
            pairs.append(comb)

    print("Generated pair candidates")

    for pair in pairs:
        for elem in set_list:
            if pair.issubset(elem):
                pair_dict[pair] += 1

    return pair_dict


def gen_counted_tuples(previous_iteration, min_support, set_list):
    k = 3

    candidate_tuples = []
    candidate_dict = defaultdict()

    current_tuples = previous_iteration.keys()

    while len(current_tuples) != 0:
        tuple_generator = gen_candidate_k_tuple(current_tuples, k, min_support)

        for comb in tuple_generator:
            if comb not in candidate_tuples:
                # if k > 3:
                #     print(comb)
                candidate_tuples.append(comb)

        for c_tuple in candidate_tuples:
            for elem in set_list:
                if c_tuple.issubset(elem):
                    candidate_dict[c_tuple] += 1

        freq_tuples = dict(
            filter(lambda elem: elem[1] >= min_support, candidate_dict.items())
        )

        if len(freq_tuples) > 0:
            yield (freq_tuples)

        current_tuples = freq_tuples
        candidate_tuples.clear()
        candidate_dict.clear()

        k += 1


def main():
    args = arg_parser.parse_args()

    if args.testfile:
        create_testfile(args.dataset, args.chunksize)
        exit()

    author_set_list = []
    freq_singletons = dict()
    pair_hash_dict = defaultdict()
    support = 6

    try:
        gen_entry_string = entry_string(args.dataset, args.chunksize * 1024 * 1024)
        for entry in gen_entry_string:
            tmp_author_set = create_author_set(unescape(entry))
            if len(tmp_author_set) >= 3:
                author_set_list.append(tmp_author_set)
    except FileNotFoundError:
        print(f"Could not find file {args.dataset}")

    try:
        print("Opening frequent singletons file.")
        with open("freq_singletons.pkl", "rb") as pkl_file:
            freq_singletons = pickle.load(pkl_file)
    except FileNotFoundError:
        print("Could not open frequent singletons file.")
        singletons, pair_hash_dict = count_singletons(author_set_list)
        freq_singletons = dict(
            filter(
                lambda elem: elem[1] >= support,
                singletons.items(),
            )
        )
        with open("freq_singletons.pkl", "wb") as pkl_file:
            pickle.dump(freq_singletons, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Size of author set list {len(author_set_list)}")
    print(f"Amount of freq singletons: {len(freq_singletons)}")

    freq_pairs = dict()

    try:
        print("Opening frequent pairs file.")
        with open("freq_pairs.pkl", "rb") as pkl_file:
            freq_pairs = pickle.load(pkl_file)
    except FileNotFoundError:
        print("Could not open pairs pickle file.")
        freq_pairs = dict(
            filter(
                lambda elem: elem[1] >= support,
                gen_counted_pairs(
                    list(freq_singletons.keys()),
                    author_set_list,
                    support,
                    pair_hash_dict,
                ).items(),
            )
        )
        with open("freq_pairs.pkl", "wb") as pkl_file:
            pickle.dump(freq_pairs, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"Frequent pairs: {freq_pairs}")

    # print("Generating triplets")

    tuple_generator = gen_counted_tuples(freq_pairs, support, author_set_list)
    k = 3
    max_itemset = dict()
    for freq_itemset in tuple_generator:
        print(f"{k}-tuple itemset: {freq_itemset}")
        max_itemset = freq_itemset
        k += 1

    print(f"Max frequent itemsets: {max_itemset}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
