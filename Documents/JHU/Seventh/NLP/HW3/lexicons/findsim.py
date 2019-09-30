'''
@authors Michael Lepori, Akash Chaurasia
@date 9/30/19

Completes part 1 of NLP Assignment 3.
'''

import numpy as np
import argparse

# Calculates cosine similarity, assuming two numpy vectors
def cosine_sim(vec1, vec2):
    numerator = np.sum(np.multiply(vec1, vec2))
    normalization1 = np.sqrt(np.sum(np.multiply(vec1, vec1)))
    normalization2 = np.sqrt(np.sum(np.multiply(vec2, vec2)))

    return numerator/(normalization1 * normalization2)


# Creates a hash associating a word with a vector
def parse_vectors(file):

    word_hash = {}
    first = True
    for line in file:
        if first:   # Skip metadata on first line
            first = False
            continue
        
        line_array = line.split()
        word = line_array[0]
        vec = np.array(line_array[1:]).astype(np.float)
        word_hash[word] = vec
    
    return word_hash


# Return top 10 most similar words
def find_sims(word_vec, word_hash, invalids):

    top_sims = [0] * 10
    top_words = ["[INIT]"] * 10 # Dummy, to be replaced

    # Iterate over all words in the vocab
    for candidate in word_hash.keys():
        if candidate not in invalids:
            sim = cosine_sim(word_vec, word_hash[candidate])
            # If this word is more similar than the least similar
            # word in the top 10, replace.
            if sim > top_sims[0]:
                top_sims[0] = sim
                top_words[0] = candidate
                # Sort top 10 so least similar word is in the first place
                top_sims, top_words = (list(t) for t in zip(*sorted(zip(top_sims, top_words))))

    return top_words


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("lexicon_file")
    parser.add_argument("word")
    parser.add_argument("--plus", default="[NONE]")
    parser.add_argument("--minus", default="[NONE]")

    args = parser.parse_args()

    # Read in file, parse it
    f = open(args.lexicon_file, 'r')
    word_hash = parse_vectors(f)
    word = args.word
    invalids = [word]

    # Handle vector additions and subtractions for analogies.
    word_vec = word_hash[word]
    if args.plus != "[NONE]":
        word_vec += word_hash[args.plus]
        invalids.append(args.plus)
    if args.minus != "[NONE]":
        word_vec -= word_hash[args.minus]
        invalids.append(args.minus)

    # Calculate sims
    sims_list = find_sims(word_vec, word_hash, invalids)

    # Print them
    for word in sims_list:
        print(word)