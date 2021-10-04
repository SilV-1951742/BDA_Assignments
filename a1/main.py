from functools import partial
import re
from typing import Dict
import xml.etree.ElementTree as ET
from html import unescape
import argparse
import os
import pickle
import itertools
import hashlib

buckets_count = 1000
support = 4
k = 2

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
    author_set = set()
    for author in re.finditer(author_regex, xml_string):
        author_set.add(author.group(1))
    return frozenset(author_set)


def create_testfile(dataset: str, chunksize: int):
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


# Hashes a given tuple to a (finite) bucket id
# TODO: fix hasher not being deterministic. Perhaps the .encode method is not deterministic?
def tuple_hasher(tuple):
    hash = hashlib.sha256(int(str(tuple).encode('utf-8')))
    bucket_id = int.from_bytes(hash.digest(), 'big') % buckets_count
    print("hashed " + str(tuple) + " to: " + str(bucket_id))
    return bucket_id


def pcy_first_pass(set_list):
    singleton_dict = dict()
    buckets = dict()
    for set in set_list:
        for elem in set:
            if elem in singleton_dict:
                singleton_dict[elem] += 1
            else:
                singleton_dict[elem] = 1
        if (len(set) > 1):
            pairs = list(itertools.combinations(set, 2))
            for pair in pairs:
                bucket = tuple_hasher(pair)
                if bucket in buckets:
                    buckets[bucket] += 1
                else:
                    buckets[bucket] = 1

    frequent_singletons = list(
        dict(
            filter(lambda elem: elem[1] >= support, singleton_dict.items())
        ).keys()
    )

    with open("frequent_singletons.pkl", "wb") as pkl_file:
        pickle.dump(frequent_singletons, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Convert buckets to bitvector
    bitvector = []
    for bucket in range(0, buckets_count):
        if bucket in buckets:
            if (buckets[bucket] > support):
                bitvector.append(True)
            else:
                bitvector.append(False)
        else:
            bitvector.append(False)

    with open("bitvector.pkl", "wb") as pkl_file:
        pickle.dump(bitvector, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


# def pcy_second_pass(bitvector: list, frequent_singletons: list, pairs: list):
def pcy_second_pass():

    with open("frequent_singletons.pkl", "rb") as pkl_file:
        frequent_singletons = pickle.load(pkl_file)

    with open("bitvector.pkl", "rb") as pkl_file:
        bitvector = pickle.load(pkl_file)

    pairs = list(itertools.combinations(frequent_singletons, 2))


    print("Pairs before second pass: ")
    print(len(pairs))

    print("\n")

    for pair in pairs:
        bucket = tuple_hasher(pair)
        if bitvector[bucket] == True:
            for item in pair:
                if (item not in frequent_singletons):
                    pairs.remove(pair)
                    break  
        else:
            pairs.remove(pair)

    print("Resulting possible frequent pairs after second pass: " )
    print(len(pairs))


def main():
    args = arg_parser.parse_args()

    if args.testfile:
        create_testfile(args.dataset, args.chunksize)
        exit()

    author_set_list = []

    try:
        gen_entry_string = entry_string(args.dataset, args.chunksize * 1024 * 1024)
        for entry in gen_entry_string:
            tmp_entry = unescape(entry)
            author_set_list.append(create_author_set(tmp_entry))
    except FileNotFoundError:
        print(f"Could not find file {args.dataset}")

    pcy_first_pass(author_set_list)
    pcy_second_pass()


    # with open("singletons.pkl", "rb") as pkl_file:
    #     singleton_dict = pickle.load(pkl_file)
    #     # Generate frequent singletons
    #     freq_singletons = dict(
    #         filter(lambda elem: elem[1] >= support, singleton_dict.items())
    #     )

"""
    # Generate candidate pairs    
    candidate_pairs = list(itertools.combinations(freq_singletons.keys(), 2))

    k_tuples_dict = dict()

    for tuple in candidate_pairs:
        for author_set in author_set_list:
            if (len(author_set.intersection(tuple)) == k):
                frozen_tuple = frozenset(tuple)
                if frozen_tuple in k_tuples_dict:
                    k_tuples_dict[frozen_tuple] += 1
                else:
                    k_tuples_dict[frozen_tuple] = 1
    
    k_tuples_dict = dict(
        filter(lambda elem: elem[1] >= support, k_tuples_dict.items())
    )
"""

    # print(k_tuples_dict)
    


    # for item in k_tuples_dict:
    #     print(item)


    # print(freq_items)

    # k = 2

    # k_freq_items_combinations = list(itertools.combinations(freq_items.keys(), 2))
    
    # print(k_freq_items_combinations)

    # print(k_tuples)

    # for current_k in range(2,k+1):

    #     print(current_k)

    #     k_tuples = list(itertools.combinations(freq_k_tuples.keys(), current_k))

    #     k_tuples_dict = dict()

    #     for tuple in k_tuples:
    #         for author_set in author_set_list:
    #             if (len(author_set.intersection(tuple)) == current_k):
    #                 frozen_tuple = frozenset(tuple)
    #                 if frozen_tuple in k_tuples_dict:
    #                     k_tuples_dict[frozen_tuple] += 1
    #                 else:
    #                     k_tuples_dict[frozen_tuple] = 1

    #     freq_k_tuples = dict(
    #         filter(lambda elem: elem[1] >= support, k_tuples_dict.items())
    #     )

    #     print(freq_k_tuples)

    #     for item in freq_k_tuples:
    #         print(item)

    # for item in k_tuples:
    #     print(k_tuples)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
