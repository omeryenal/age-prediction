"""
Assignment 2 – Filter Long Names

Write a function `filter_long_names(name_list, min_length)` that returns a list of names longer than `min_length`.

Parameters:
- name_list: a list of strings
- min_length: integer value (default = 5)

Example:
filter_long_names(["Ali", "Zeynep", "Can"], 4) -> ["Zeynep"]
"""
def filter_long_names(name_list, min_length):
    for word in name_list:
        if min_length > len(word):
            name_list.remove(word)
    return name_list

print(filter_long_names(["Ali", "Zeynep", "Can"], 4))