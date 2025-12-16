import string
import sys
from collections import deque

"""
file: hw1.py
CSCI-331
author: Matthew Morrison msm8275

Find the shortest path from a start word and end word using a dictionary of
words, altering one letter at a time.
"""

def recreate_path(path, start_word, end_word):
    """
    Create the final shortest path from start_word to end_word. This will print
    out line by line each word used utilizing a stack
    :param path: a dictionary with child nodes as keys and parent nodes
    as values
    :param start_word: the original word
    :param end_word: the final word to end on
    """
    final_path = [end_word]
    while end_word != start_word:
        curr_word = path[end_word]
        end_word = curr_word
        final_path.append(curr_word)

    while final_path:
        print(final_path.pop())


def search_for_word_chain_path(file_path, start_word, end_word):
    """
    Use a modified BFS algorithm to find the shortest path from start_word
    to end_word. Search through each depth using a queue to discover the
    path to each word. Will print out 'No solution' if words are not of
    equal length or if the end word does not exist in the word list
    :param file_path: file path to the list of valid words
    :param start_word: the word to start searching from
    :param end_word: the final word to end on
    """
    words = set()
    path = dict() # hold child node as key, parent as value

    # open file path and add all words to a set
    with open(file_path) as file:
        for line in file:
            word = line.strip()
            words.add(word)

    # check if end word is even in the set
    if end_word not in words or len(end_word) != len(start_word):
        print("No solution")
        return

    visited_words = {start_word} # to avoid infinite loops
    queue = deque([start_word])
    word_length = len(start_word)

    while queue:
        current_word = queue.popleft()
        for i in range(word_length):
            for c in string.ascii_lowercase:
                # go through each ascii char and check if it is a valid word
                # add to queue if valid and not already visited
                new_word = current_word[:i] + c + current_word[i + 1:]

                if new_word in words and new_word not in visited_words:
                    queue.append(new_word)
                    visited_words.add(new_word)
                    # set child as key and parent as value
                    path[new_word] = current_word

                # recreate path once end word has been discovered
                if new_word == end_word:
                    recreate_path(path, start_word, end_word)
                    return
    # ran out of valid words, no path is found
    print("No solution")
    return

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 hw1.py <word_list> <start_word> <end_word>")
        return

    file_path = sys.argv[1]
    start_word = sys.argv[2]
    end_word = sys.argv[3]

    search_for_word_chain_path(file_path, start_word, end_word)


if __name__ == '__main__':
    main()