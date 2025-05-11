import unittest
from assignment1_is_prime import is_prime
from assignment2_long_names import filter_long_names
from assignment3_word_freq import word_frequencies
from assignment4_zip_dict import create_dict
from assignment5_file_io import save_numbers_to_file, load_numbers_from_file
import os

class TestDay01Assignments(unittest.TestCase):

    def test_is_prime(self):
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(13))
        self.assertFalse(is_prime(100))

    def test_filter_long_names(self):
        names = ["Ali", "Zeynep", "Can", "Mustafa", "Ada"]
        result = filter_long_names(names, min_length=4)
        self.assertEqual(result, ["Zeynep", "Mustafa"])

    def test_word_frequencies(self):
        text = "Hello, hello! World. world?"
        result = word_frequencies(text)
        expected = {"hello": 2, "world": 2}
        self.assertEqual(result, expected)

    def test_create_dict(self):
        keys = ["a", "b", "c"]
        values = [1, 2, 3]
        result = create_dict(keys, values)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

        with self.assertRaises(ValueError):
            create_dict(["a", "b"], [1])  # Unequal lengths

    def test_file_io(self):
        filename = "test_numbers.txt"
        numbers = [10, 20, 30, 40]
        save_numbers_to_file(numbers, filename)

        loaded = load_numbers_from_file(filename)
        self.assertEqual(loaded, numbers)

        os.remove(filename)

if __name__ == "__main__":
    unittest.main()
