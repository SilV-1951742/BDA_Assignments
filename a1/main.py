from functools import partial
import re
from html import unescape
import argparse
import os
import itertools
import hashlib
from typing import Final
from collections import defaultdict

HASH_BUCKETS: Final = 500000

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
    help="Create a test dataset of x entries.",
)


def hash_tuple(tuple_object):
    tuple_bytes = bytearray(str(tuple_object), "utf-8")
    return int(hashlib.sha256(tuple_bytes).hexdigest(), 16) % HASH_BUCKETS


def entry_string(filename: str, chunk_size: int):
#def entry_string(filename, chunk_size):
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
#def create_author_set(xml_string):
    """
    This function takes an entry and returns the author set.
    """
    author_set = set()
    for author in re.finditer(author_regex, xml_string):
        author_set.add(author.group(1))
    return frozenset(author_set)


def create_testfile(dataset: str, chunksize: int):
#def create_testfile(dataset, chunksize):
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

        for _ in range(150000):
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

    # def gen_counted_tuples(previous_iteration, min_support, set_list):
    #     k = 3

    #     candidate_tuples = []
    #     candidate_dict = defaultdict()

    #     current_tuples = previous_iteration.keys()

    #     while len(current_tuples) != 0:
    #         tuple_generator = gen_candidate_k_tuple(current_tuples, k, min_support)

    #         for comb in tuple_generator:
    #             if comb not in candidate_tuples:
    #                 # if k > 3:
    #                 #     print(comb)
    #                 candidate_tuples.append(comb)

    # def expand_sets(previous_sets, k):
    #     # Calculate occuring singletons in previous sets and form array
    #     singletons = []
    #     for previous_set in previous_sets:
    #         for elem in previous_set:
    #             if elem not in singletons:
    #                 singletons.append(elem)

    #     # Expand all sets from previous sets with a singleton
    #     new_k_sets = []
    #     for previous_set in previous_sets:
    #         for singleton in singletons:
    #             new_k_set = list(previous_set)
    #             new_k_set.append(singleton)
    #             new_k_set = frozenset(new_k_set)
    #             if new_k_set not in new_k_sets and len(new_k_set) == k:
    #                 new_k_sets.append(new_k_set)

    #     # print(new_k_sets)

    #     return new_k_sets

    # def gen_candidate_k_tuple(singletons, min_support, k, previous_iteration = dict()):
    #     """
    #     Generator to calculate k sized tuples.
    #     """
    #     for comb in itertools.combinations(singletons, k):
    #         for elem in comb:
    #             if singletons[elem] < min_support:
    #                 continue
    #             if len(previous_iteration) > 0:
    #                 print("Do something")
    #         yield frozenset(comb)

    # def gen_counted_pairs(singletons, set_list, support):
    """
    Function that creates a dictionary of counted pairs.
    """
    # sets = singletons
    # k_sets_not_empty = True
    # current_k = 2

    # while (k_sets_not_empty):

    #     print("Current k: " + str(current_k))

    #     if os.path.exists("sets_" + str(current_k) + ".pkl"):
    #         print("Opening " + str(current_k) + " pairs pickle file.")
    #         with open("sets_" + str(current_k) + ".pkl", "rb") as pkl_file:
    #             set_dict = pickle.load(pkl_file)
    #     else:
    #         # Generate sets of size k
    #         sets = expand_sets(sets, current_k)

    #         print("Generated " + str(current_k) + "-set candidates")

    #         # print(set_list)
    #         # print("\n\n\n")

    #         set_dict = dict()
    #         for set in sets:
    #             for elem in set_list:
    #                 if set.issubset(elem):
    #                     # print(f"Found matching pair {pair} in {elem}")
    #                     if set in set_dict:
    #                         set_dict[set] += 1
    #                     else:
    #                         set_dict[set] = 1

    #         if (len(set_dict) == 0):
    #             k_sets_not_empty = False

    #         set_dict = {key:value for (key, value) in set_dict.items() if value >= support}

    #         # set_dict = dict(
    #         #     filter(
    #         #         lambda elem: elem[1] >= support,
    #         #         # gen_counted_pairs(freq_singletons, 4, author_set_list).items()
    #         #         set_dict.items()
    #         #     )
    #         # )

    #         print(set_dict)

    #         # print(freq_k_sets_dict)
    #         print("Amount of frequent " + str(current_k) + "-sized author groups found: " + str(len(set_dict)))

    #         with open("sets_" + str(current_k) + ".pkl", "wb") as pkl_file:
    #             pickle.dump(set_dict, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    #         sets = list(set_dict.keys())
    #         current_k += 1


def gen_candidate_k_tuple(previous_iteration, k: int, min_support, pair_hash=dict()):
#def gen_candidate_k_tuple(previous_iteration, k, min_support, pair_hash=dict()):
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
    pair_dict = dict()

    pair_generator = gen_candidate_k_tuple(singletons, 2, min_support, pair_hash)
    for comb in pair_generator:
        if comb not in pairs:
            pairs.append(comb)

    print("Generated pair candidates")

    for pair in pairs:
        for elem in set_list:
            if pair.issubset(elem):
                if pair not in pair_dict:
                    pair_dict[pair] = 1
                else:
                    pair_dict[pair] += 1

    return pair_dict


def gen_counted_tuples(previous_iteration, min_support, set_list):
    k = 3

    candidate_tuples = []
    candidate_dict = dict()

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
                    if c_tuple not in candidate_dict:
                        candidate_dict[c_tuple] = 1
                    else:
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
    # <<<<<<< HEAD
    pair_hash_dict = dict()
    support = 6
    # =======
    #     support = 15
    #     k = 3
    # >>>>>>> 0c92ff29e8e3f7efd4753eca0d3b85b9c6d96139

    try:
        gen_entry_string = entry_string(args.dataset, args.chunksize * 1024 * 1024)
        for entry in gen_entry_string:
            tmp_author_set = create_author_set(unescape(entry))
            if len(tmp_author_set) >= 3:
                author_set_list.append(tmp_author_set)
    except FileNotFoundError:
        print(f"Could not find file {args.dataset}")

    singletons, pair_hash_dict = count_singletons(author_set_list)
    freq_singletons = dict(
        filter(
            lambda elem: elem[1] >= support,
            singletons.items(),
        )
    )

    print(f"Size of author set list {len(author_set_list)}")
    print(f"Amount of frequent singletons: {len(freq_singletons)}")

    freq_pairs = dict()

    
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

    print(f"Frequent pairs: {freq_pairs}")
    print()
    print()

    tuple_generator = gen_counted_tuples(freq_pairs, support, author_set_list)
    k = 3
    max_itemset = dict()
    for freq_itemset in tuple_generator:
        print(f"{k}-tuple itemset: {freq_itemset}")
        print()
        print()
        max_itemset = freq_itemset
        k += 1

    print(f"Max frequent itemsets: {max_itemset}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
