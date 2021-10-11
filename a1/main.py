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
        for _ in range(50000):
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


def calculateMaxFrequentItemsets(countedOccurrences):
    maxFrequentItemsets = dict(
        filter(
            lambda elem: elem[1] == max(countedOccurrences.values()), countedOccurrences.items()
        )
    )
    return maxFrequentItemsets


def countOccurrences(combinations, author_set_list):
    countedOccurrences = dict()
    for authorSet in author_set_list:
        for combination in combinations:
            if authorSet.issubset(combination):
                if combination in countedOccurrences:
                    countedOccurrences[combination] += 1
                else:
                    countedOccurrences[combination] = 1
    return countedOccurrences


def calculateSupportedAuthorSets(authorSets, support):
    return dict(
        filter(
            lambda elem: elem[1] >= support,
            authorSets.items(),
        )
    )


def makeCombinations(author_set, k):
    combinations = []
    valid_combinations = itertools.combinations(author_set, k)
    for combination in valid_combinations:
        if (len(combination) == k):
            combinations.append(frozenset(combination))
    return combinations


def main():
    args = arg_parser.parse_args()

    if args.testfile:
        create_testfile(args.dataset, args.chunksize)
        exit()

    author_set_list = []
    support = 3

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


    print("Max frequent itemsets, size: 1: ", end='')
    print(calculateMaxFrequentItemsets(freq_singletons))

    k = 2

    while True:
        combinations = dict()
        for author_set in author_set_list:
            if len(author_set) >= k:
                for combination in makeCombinations(author_set, k):
                    if combination in combinations:
                        combinations[combination] += 1
                    else:
                        combinations[combination] = 1

        supportedAuthorSets = calculateSupportedAuthorSets(combinations, support)
        
        print("Max frequent itemsets, size " + str(k) + " :" , end='')
        print(calculateMaxFrequentItemsets(supportedAuthorSets))

        k += 1

        if len(supportedAuthorSets) == 0:
            break

    print("done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
