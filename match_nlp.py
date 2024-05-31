import pickle
from fuzzywuzzy import fuzz
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from scipy.stats import ttest_ind
import numpy as np
from textstat import flesch_kincaid_grade, flesch_reading_ease

nltk.download('punkt')


def calculate_pmi(text):
    tokens = nltk.word_tokenize(text.lower())
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_freq = bigram_finder.ngram_fd
    word_freq = FreqDist(tokens)
    total_bigrams = bigram_freq.N()
    total_words = word_freq.N()

    pmi_scores = []
    for bigram, freq in bigram_freq.items():
        word1, word2 = bigram
        p_word1 = word_freq[word1] / total_words
        p_word2 = word_freq[word2] / total_words
        p_word1_word2 = freq / total_bigrams
        pmi = np.log2(p_word1_word2 / (p_word1 * p_word2))
        pmi_scores.append((bigram, pmi))

    return sorted(pmi_scores, key=lambda x: -x[1])


def compute_statistics(sample1, sample2):
    mean1, variance1 = np.mean(sample1), np.var(sample1)
    mean2, variance2 = np.mean(sample2), np.var(sample2)
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False)
    return mean1, variance1, mean2, variance2, t_stat, p_value


def readability_scores(text):
    fk_score = flesch_kincaid_grade(text)
    fr_score = flesch_reading_ease(text)
    return fk_score, fr_score


def find_matching_images(keyword):
    # Load the dictionary from the pickle file
    with open("label.pickle", "rb") as f:
        label_dict = pickle.load(f)

    # Calculate the fuzzy matching ratio
    match_dict = {word: fuzz.partial_ratio(keyword.lower(), word.lower()) for word in label_dict}

    # Find the word with the highest match ratio
    matched_word = max(match_dict, key=match_dict.get)

    # Combine all descriptions or related texts for the matched word
    combined_text = " ".join(label_dict[matched_word])

    # Calculate PMI scores for the matched descriptions
    pmi_scores = calculate_pmi(combined_text)

    # Calculate readability scores
    fk_score, fr_score = readability_scores(combined_text)

    # Compute statistics between matching ratios
    sample1 = list(match_dict.values())
    sample2 = [ratio for word, ratio in match_dict.items() if word != matched_word]
    mean1, variance1, mean2, variance2, t_stat, p_value = compute_statistics(sample1, sample2)

    return {
        'matched_word': matched_word,
        'ratio': match_dict[matched_word],
        'animal_list': label_dict[matched_word],
        'pmi_scores': pmi_scores,
        'readability': {'fk_score': fk_score, 'fr_score': fr_score},
        'stats': {
            'mean1': mean1,
            'variance1': variance1,
            'mean2': mean2,
            'variance2': variance2,
            't_stat': t_stat,
            'p_value': p_value
        }
    }


if __name__ == "__main__":
    keyword = "your_keyword_here"
    results = find_matching_images(keyword)
    print(results)
