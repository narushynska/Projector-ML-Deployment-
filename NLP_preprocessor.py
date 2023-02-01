import re
import string

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin


def download_if_non_existent(res_path, res_name):
  try:
    nltk.data.find(res_path)
  except LookupError:
    print(f'resource {res_path} not found. Downloading now...')
    nltk.download(res_name)

def lemmatize_pos_tagged_text(text, lemmatizer, post_tag_dict):
  sentences = nltk.sent_tokenize(text)
  new_sentences = []

  for sentence in sentences:
    sentence = sentence.lower()
    new_sentence_words = []
    #one pos_tuple for sentence
    pos_tuples = nltk.pos_tag(nltk.word_tokenize(sentence)) 

    for word_idx, word in enumerate(nltk.word_tokenize(sentence)):
      nltk_word_pos = pos_tuples[word_idx][1]
      wordnet_word_pos = post_tag_dict.get(
                          nltk_word_pos[0].upper(), None)
      if wordnet_word_pos is not None:
        new_word = lemmatizer.lemmatize(word, wordnet_word_pos)
      else:
        new_word = lemmatizer.lemmatize(word)

      new_sentence_words.append(new_word)

    new_sentence = " ".join(new_sentence_words)
    new_sentences.append(new_sentence)

  return " ".join(new_sentences)

class NltkPreprocessingSteps:
  def __init__(self, X):
    self.X = X
    download_if_non_existent('corpora/stopwords', 'stopwords')
    download_if_non_existent('tokenizers/punkt', 'punkt')
    download_if_non_existent('taggers/averaged_perceptron_tagger',
                             'averaged_perceptron_tagger')
    download_if_non_existent('corpora/wordnet', 'wordnet')

    self.sw_nltk = stopwords.words('english')
    
    self.post_tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

    self.remove_punctuations = string.punctuation.replace('.','')

  def remove_numbers(self):
    self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
    return self

  def remove_punctuations_except_periods(self):
    self.X = self.X.apply(
                 lambda x: re.sub('[%s]' %
                  re.escape(self.remove_punctuations), '' , x))
    return self

  def remove_all_punctuations(self):
    self.X = self.X.apply(lambda x: re.sub('[%s]' %
                          re.escape(string.punctuation), '' , x))
    return self

  def remove_double_spaces(self):
    self.X = self.X.apply(lambda x: re.sub(' +', ' ', x))
    return self

  def remove_stopwords(self):
    # remove stop words from token list in each column
    self.X = self.X.apply(
            lambda x: " ".join([ word for word in x.split() 
                     if word not in self.sw_nltk]) )
    return self

  def lemmatize(self):
    lemmatizer = WordNetLemmatizer()
    self.X = self.X.apply(lambda x: lemmatize_pos_tagged_text(
                           x, lemmatizer, self.post_tag_dict))
    return self

  def get_processed_text(self):
    return self.X


class NltkTextPreprocessor(TransformerMixin, BaseEstimator):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    txt_preproc = NltkPreprocessingSteps(X.copy())
    processed_text = \
            txt_preproc \
            .remove_numbers()\
            .remove_stopwords()\
            .remove_punctuations_except_periods()\
            .lemmatize()\
            .remove_double_spaces()\
            .remove_all_punctuations()\
            .get_processed_text()
    return processed_text