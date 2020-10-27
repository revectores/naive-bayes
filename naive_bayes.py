import re
from collections import Counter
from itertools import chain
from math import log
from pprint import pprint


from nltk.corpus import movie_reviews, stopwords, wordnet
from nltk.corpus import stopwords
import nltk.stem as ns
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from nltk import pos_tag


# Tokenization (already done by nltk) and normalization
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

tokens_list, categories = zip(*documents)

pattern = "[^a-zA-Z0-9\n ]"
tokens_list = [[re.sub(pattern, "", token).lower() for token in tokens if re.sub(pattern, "", token)] for tokens in
               tokens_list]
tokens_list = [[token for token in tokens if (token not in stopwords.words('english'))] for tokens in tokens_list]


# Stemming
token_stems_list = [[PorterStemmer().stem(token) for token in tokens] for tokens in tokens_list]


# Lemmatization
token_tags_list = [pos_tag(tokens) for tokens in token_stems_list]

tag_m = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'R': wordnet.ADV
}
tag_map = defaultdict(lambda: wordnet.NOUN, tag_m)

token_stems_list = [[ns.WordNetLemmatizer().lemmatize(token, pos=tag_map[postag[0]])
                     for token, postag in token_tags] for token_tags in token_tags_list]


# Bayes Algorithm
words_list = list(chain.from_iterable(token_stems_list))
words_list_neg = list(chain.from_iterable(token_stems_list[:800]))
words_list_pos = list(chain.from_iterable(token_stems_list[1000:1800]))
words_set = set(words_list)
words_set_neg = set(words_list_neg)
words_set_pos = set(words_list_pos)

word_count_neg = Counter(words_list_neg)
word_count_pos = Counter(words_list_pos)

neg_words_count = len(words_list_neg)
pos_words_count = len(words_list_pos)
total_unique_words_count = len(words_set)

total_neg = neg_words_count + total_unique_words_count
smoothing_freq_neg = log(1 / total_neg)
word_freq_neg = {
    word: log((count + 1) / total_neg)
    for word, count in word_count_neg.items()
}

total_pos = pos_words_count + total_unique_words_count
smoothing_freq_pos = log(1 / total_pos)
word_freq_pos = {
    word: log((count + 1) / total_pos)
    for word, count in word_count_pos.items()
}

word_freq_neg.update(
    dict.fromkeys(words_set - words_set_neg, smoothing_freq_neg)
)
word_freq_pos.update(
    dict.fromkeys(words_set - words_set_pos, smoothing_freq_pos)
)

training_comments = token_stems_list[800:1000] + token_stems_list[1800:]
training_categories = categories[800:1000] + categories[1800:]
confusion_matrix = [[0, 0], [0, 0]]

for index, comment in enumerate(training_comments):
    neg_prob = 0
    pos_prob = 0

    for word in comment:
        neg_prob += word_freq_neg[word]
        pos_prob += word_freq_pos[word]

    actual = training_categories[index]
    predict = 'neg' if neg_prob > pos_prob else 'pos'
    confusion_matrix[predict == 'neg'][actual == 'neg'] += 1

tp, fp = confusion_matrix[0]
fn, tn = confusion_matrix[1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * precision * recall) / (precision + recall)

print("""Accuracy = {0:.2f}
Precision = {1:.2f}
Recall = {2:.2f}
F1 = {3:.2f}
""".format(accuracy, precision, recall, f1))


