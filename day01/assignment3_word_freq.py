"""
Assignment 3 – Word Frequency Counter

Write a function `word_frequencies(text)` that returns a dictionary of word counts in the given string.

Requirements:
- Case-insensitive
- Strip punctuation (.,!?)

Example:
word_frequencies("Hello hello world!") -> {"hello": 2, "world": 1}
"""
def word_frequencies(text):
    freq = {}
    for word in text.lower().split():
        word = word.strip('.,!?')
        freq[word] = freq.get(word, 0) + 1
    return freq