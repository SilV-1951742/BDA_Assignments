from functools import partial
import re
from html import unescape
import argparse
import os
import itertools
import numpy as np
import hashlib
from typing import Final


HASH_BUCKETS: Final = 1000000

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


def hash_set(set_object):
    set_bytes = bytearray(str(set_object), "utf-8")
    return int(hashlib.sha256(set_bytes).hexdigest(), 16) % HASH_BUCKETS


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
        for _ in range(1500000):
            tmp_entry = unescape(next(gen_entry_string))
            author_set_list.append(create_author_set(tmp_entry))
            f.write(tmp_entry)


def count_singletons(set_list):
    """
    This function takes a list of author sets and creates a picklefile
    with a dictionary with the counted singleton authors.
    """
    singleton_dict = dict()
    hash_dict = dict()
    
    for sets in set_list:
        for elem in sets:
            if elem in singleton_dict:
                singleton_dict[elem] += 1
            else:
                singleton_dict[elem] = 1
        for comb in itertools.combinations(sets, 2):
            pair_hash = hash_set(comb)
            if pair_hash not in hash_dict:
                hash_dict[pair_hash] = 1
            else:
                hash_dict[pair_hash] += 1

    return singleton_dict, hash_dict


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


def validSet(previous_iteration, k: int, candidate):
    for min_combo in itertools.combinations(candidate, k - 1):
        if frozenset(min_combo) not in previous_iteration:
            return False
    return True


def generateBigKCandidate(previous_iteration, k: int):
    """
    Generates k-set candidates, more memory efficient but slower for big k values
    """
    for set_pair in itertools.combinations(previous_iteration, 2):
        for combination in itertools.combinations(set_pair[0].union(set_pair[1]), k):
            if validSet(previous_iteration, k, combination) == True:
                yield frozenset(combination)

            
def makeCombinationsBigK(previous_iteration, k: int):
    combinations = []
    np_pi = np.array(list(previous_iteration))
    for combination in generateBigKCandidate(np_pi, k):
        if combination not in combinations:
            combinations.append(combination)

    return combinations



def main():
    args = arg_parser.parse_args()

    if args.testfile:
        create_testfile(args.dataset, args.chunksize)
        exit()

    author_set_list = []
    #current_hash_dict = dict()
    support = 25

    try:
        gen_entry_string = entry_string(args.dataset, args.chunksize * 1024 * 1024)
        for entry in gen_entry_string:
            tmp_author_set = create_author_set(unescape(entry))
            if len(tmp_author_set) > 0:
                author_set_list.append(tmp_author_set)
    except FileNotFoundError:
        print(f"Could not find file {args.dataset}")

    print("Dataset loaded.")

    singletons, _ = count_singletons(author_set_list)
    freq_singletons = dict(
        filter(
            lambda elem: elem[1] >= support,
            singletons.items(),
        )
    )

    np_author_set_list = np.array(author_set_list)

    print(f"Size of author set list {len(np_author_set_list)}")
    print(f"Amount of frequent singletons: {len(freq_singletons)}")

    # print("Max frequent itemsets, size 1: ", end='')
    # print(calculateMaxFrequentItemsets(freq_singletons))

    k = 2

    supportedAuthorSets = dict()
    maximal_f_itemsets = dict()
    k_res = 0
    
    while k <= 3:
        combinations = dict()

        # next_hash_dict = dict()
        
        for author_set in np_author_set_list:
            if len(author_set) >= k:
                for combination in makeCombinations(author_set, k):
                    try:
                        #if current_hash_dict[hash_set(combination)] > support:
                        if combination in combinations:
                            combinations[combination] += 1
                        else:
                            combinations[combination] = 1
                    except KeyError:
                        continue
                    
                # if len(author_set) > k and k < 3:
                #     for combination in makeCombinations(author_set, k + 1):
                #         comb_hash = hash_set(combination)
                #         if comb_hash in next_hash_dict:
                #             next_hash_dict[comb_hash] += 1
                #         else:
                #             next_hash_dict[comb_hash] = 1

        # current_hash_dict = next_hash_dict.copy()
        # next_hash_dict.clear()
        supportedAuthorSets = calculateSupportedAuthorSets(combinations, support)

        if(supportedAuthorSets):
            maximal_f_itemsets = supportedAuthorSets
            k_res = k
        
        print(f"{k} itemsets: " , end='')
        print(calculateMaxFrequentItemsets(supportedAuthorSets))

        k += 1

        if len(supportedAuthorSets) == 0:
            break

        
    while supportedAuthorSets:
        combinations = dict()
        candidateSets = makeCombinationsBigK(supportedAuthorSets.keys(), k)
        print("Generated candidates")
        for author_set in np_author_set_list:
            if len(author_set) >= k:
                for candidate in candidateSets:
                    if candidate.issubset(author_set):
                        if candidate in combinations:
                            combinations[candidate] += 1 
                        else:
                            combinations[candidate] = 1
        print("Counted candidates")
        supportedAuthorSets = calculateSupportedAuthorSets(combinations, support)

        if(supportedAuthorSets):
            maximal_f_itemsets = supportedAuthorSets
            k_res = k
        #print(supportedAuthorSets)
        print(f"{k} itemsets: " , end='')
        print(calculateMaxFrequentItemsets(supportedAuthorSets))
        
        k += 1
        
        # if len(supportedAuthorSets) == 0:
        #     break

    print(f"The maximal frequent items sets are {k_res}-sets:")
    print(maximal_f_itemsets)

    print("done")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
