Write a program to solve the following search problem: You are given two words of equal length and a dictionary of legal English words. At each step, you can change any single letter in the word to any other letter, provided that the result is a word in the dictionary; you cannot add or remove letters. Your program should print the shortest list of words that connects the two given words in this way (if there are multiple such paths, any one is sufficient).

It prints a chaing of words that leads from the start word to the target word OR "No solution" if it is impossible to find a path

Usage:  
python hw1.py [word_dictionary] [word1] [word2]

Example output:  
python hw1.py /usr/share/dict/words harp ramp  
harp  
carp  
camp  
ramp
