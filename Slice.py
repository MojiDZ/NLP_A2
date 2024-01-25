import string
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from math import sqrt
from collections import defaultdict


def text_normalization(text: str) -> list[str]:
    """
    Perform text normalization by removing punctuation, tokenizing, removing stop words, and stemming.
    """
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()

    # Remove punctuation
    text = remove_punctuation(text)

    # Tokenization and filtering out short tokens
    tokens = [token.lower() for token in word_tokenize(text) if len(token) > 1]

    # Stop words removal
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stems = [porter_stemmer.stem(token) for token in tokens]

    return stems


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the input text."""
    punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
    return punctuation_regex.sub("", text)


def create_bag_of_words(text: str) -> dict:
    """
    Create a bag of words from the input text.
    """
    word_frequencies = defaultdict(int)

    normalized_words = text_normalization(text)

    for word in normalized_words:
        word_frequencies[word] += 1

    return dict(word_frequencies)


def normalize_bag_of_words(pool: dict, _slice: dict) -> dict:
    """
    Normalize a bag of words based on the given pool and slice.
    """
    total_length = sum(pool.values())

    normalized_bow = {}

    for word in pool:
        slice_occurrence = _slice.get(word, 0)
        normalized_bow[word] = slice_occurrence / total_length

    return normalized_bow


def does_sentence_fit(sentence: str, word_slice: list[str], window_size: int) -> bool:
    """
    Check if a sentence fits within the given window size when combined with a word slice.
    """
    combined_tokens_length = len(word_tokenize(' '.join(word_slice))) + len(word_tokenize(sentence))
    return combined_tokens_length <= window_size


def calculate_squared_sum(vector: list[float]) -> float:
    """
    Calculate the squared sum of elements in the given vector.
    """
    squared_sum_result = round(sqrt(sum(a * a for a in vector)), 3)
    return squared_sum_result


def cosine_similarity(pool: dict, slice1: str, slice2: str) -> float:
    """
    Calculate the cosine similarity between two slices based on a given pool.
    """
    normalized_slice1 = normalize_bag_of_words(pool, create_bag_of_words(slice1))
    normalized_slice2 = normalize_bag_of_words(pool, create_bag_of_words(slice2))

    values1 = list(normalized_slice1.values())
    values2 = list(normalized_slice2.values())

    numerator = sum(a * b for a, b in zip(values1, values2))
    denominator = calculate_squared_sum(values1) * calculate_squared_sum(values2)

    return round(numerator / float(denominator), 3)


class Slice:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            self.content: str = file.read()  # content of the file
            self.sentences: list[str] = sent_tokenize(self.content)  # list of sentences in the text
            self.token_count: int = len(word_tokenize(self.content))  # total number of tokens in the text
            self.word_pool: dict = create_bag_of_words(self.content)  # bag of words for the text

    def slice_document(self, window_size: int) -> list[str]:
        """
        Slice the document into sections of a specified window size.
        """
        if self.token_count <= window_size:
            return [self.content]

        slices: list[str] = []
        current_slice: list[str] = []
        adjacent_slice: list[str] = []

        for sentence in self.sentences:
            if len(current_slice) == 0:
                current_slice.append(sentence)
                continue

            if len(adjacent_slice) > 0 and cosine_similarity(self.word_pool, ' '.join(adjacent_slice),
                                                             ' '.join(current_slice)) > 0.8:
                while not does_sentence_fit(sentence, current_slice, window_size):
                    current_slice.pop(0)

                current_slice.append(sentence)
                continue

            slice_token_size = len(word_tokenize(' '.join(current_slice)))
            sent_token_size = len(word_tokenize(sentence))

            if slice_token_size + sent_token_size > window_size:
                slices.append(' '.join(current_slice))
                adjacent_slice = current_slice.copy()

                while not does_sentence_fit(sentence, current_slice, window_size):
                    current_slice.pop(0)

                current_slice.append(sentence)
            else:
                current_slice.append(sentence)

        while cosine_similarity(self.word_pool, slices[-1], ' '.join(current_slice)) > 0.8:
            current_slice.pop(0)

            if len(current_slice) == 0:
                break

        if len(current_slice) != 0:
            slices.append(' '.join(current_slice))

        return slices
