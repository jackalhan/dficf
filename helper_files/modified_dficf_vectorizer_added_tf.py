# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# License: BSD 3 clause
"""
The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.
"""
from __future__ import unicode_literals

import array
import collections
import operator
from collections import Mapping, defaultdict
import numbers
from itertools import chain
from operator import itemgetter
import re
import unicodedata

import math
import numpy as np
import scipy.sparse as sp

from  sklearn.base import BaseEstimator, TransformerMixin
from  sklearn.externals import six
from  sklearn.externals.six.moves import xrange
from  sklearn.preprocessing import normalize
from  sklearn.feature_extraction.hashing import FeatureHasher
from  sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from  sklearn.utils.validation import check_is_fitted
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import spatial
import random

__all__ = ['CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'DfIcfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags']


def strip_accents_unicode(s):

    normalized = unicodedata.normalize('NFKD', s)
    if normalized == s:
        return s
    else:
        return ''.join([c for c in normalized if not unicodedata.combining(c)])


def strip_accents_ascii(s):

    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


def strip_tags(s):

    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:               # assume it's a collection
        return frozenset(stop)


class VectorizerMixin(object):
    """Provides common code for text vectorizers (tokenization logic)."""

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        """Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc

    def _word_ngrams(self, tokens, stop_words=None, lemmatizer=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if lemmatizer is not None:
            tokens_ = []
            for w in tokens:
                w_lem_n = lemmatizer.lemmatize(w)
                w_lem_v = lemmatizer.lemmatize(w, 'v')
                if w_lem_n != w:
                    w_ = w_lem_n
                elif w_lem_v != w:
                    w_ = w_lem_v
                else:
                    w_ = w
                tokens_.append(w_)
            tokens = tokens_

        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]



        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):

        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in xrange(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor


        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        if self.lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def get_stop_words(self):
        """Build or fetch the effective stop words list"""
        return _check_stop_list(self.stop_words)

    def get_lemmatizer(self):
        """Build or fetch the effective stop words list"""
        if self.stop_words == "english":
            return WordNetLemmatizer()
        else:
            return None

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            #todo : nltk root word should be embedded here
            lemmatizer = self.get_lemmatizer()
            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words, lemmatizer)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _validate_vocabulary(self):

        try:
            #TODO: try to handle multiple vocabularies.
            self.vocabularies_with_categories_ = collections.OrderedDict()
            self.fixed_vocabulary_with_categories_ = collections.OrderedDict()
            for raw_category, raw_voc in self.vocabularies_with_categories.items():
                vocabulary = raw_voc
                if vocabulary is not None:
                    if isinstance(vocabulary, set):
                        vocabulary = sorted(vocabulary)
                    if not isinstance(vocabulary, Mapping):
                        vocab = collections.OrderedDict()
                        for i, t in enumerate(vocabulary):
                            if vocab.setdefault(t, i) != i:
                                msg = "Duplicate term in vocabulary[category : %s]: %r" % (raw_category, t)
                                raise ValueError(msg)
                        vocabulary = vocab
                    else:
                        indices = set(six.itervalues(vocabulary))
                        if len(indices) != len(vocabulary):
                            msg = "[category : %s] Vocabulary contains repeated indices." % (raw_category)
                            raise ValueError(msg)
                        for i in xrange(len(vocabulary)):
                            if i not in indices:
                                msg = ("[category : %s] Vocabulary of size %d doesn't contain index "
                                       "%d." % (raw_category, len(vocabulary), i))
                                raise ValueError(msg)
                    if not vocabulary:
                        msg = "[category : %s] empty vocabulary passed to fit" % (raw_category)
                        raise ValueError(msg)
                    self.fixed_vocabulary_with_categories_[raw_category] = True
                    self.vocabularies_with_categories_[raw_category] = dict(vocabulary)

                else:
                    self.fixed_vocabulary_with_categories_[raw_category] = False
                    self.vocabularies_with_categories_[raw_category] = None
        except:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):

        """Check if vocabulary is empty or missing (not fit-ed)"""
        msg = "%(name)s - Vocabulary wasn't fitted."
        check_is_fitted(self, 'vocabularies_with_categories_', msg=msg),

        if len(self.vocabularies_with_categories_) == 0:
            raise ValueError("Vocabulary is empty")

        # """Check if vocabulary is empty or missing (not fit-ed)"""
        # for raw_category, raw_voc in self.vocabularies_with_categories.items():
        #     self.vocabulary_ = raw_voc
        #     msg = "%(name)s - Vocabulary wasn't fitted for " + raw_category
        #     check_is_fitted(self, 'vocabulary_', msg=msg),
        #
        #     if len(self.vocabulary_) == 0:
        #         msg = "Vocabulary is empty for %s" % raw_category
        #         raise ValueError(msg)


class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 non_negative=False, dtype=np.float64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.non_negative = non_negative
        self.dtype = dtype

    def partial_fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        """
        return self

    def fit(self, X, y=None):

        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._get_hasher().fit(X, y=y)
        return self

    def transform(self, X):

        if isinstance(X, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def _get_hasher(self):
        return FeatureHasher(n_features=self.n_features,
                             input_type='string', dtype=self.dtype,
                             alternate_sign=self.alternate_sign,
                             non_negative=self.non_negative)


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


class CountVectorizer(BaseEstimator, VectorizerMixin):


    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabularies_with_categories=None, binary=False, dtype=np.int64, stemming_and_lemmatization = False):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabularies_with_categories = vocabularies_with_categories
        self.binary = binary
        self.dtype = dtype
        self.stemming_and_lemmatization = stemming_and_lemmatization

    def _sort_features(self, X, vocabulary):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def _limit_features(self, X, vocabulary, high=None, low=None,
                        limit=None):

        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms

    def _count_vocab(self, raw_documents_with_categories, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        category_voc_dict = collections.OrderedDict()
        for raw_category, raw_documents in raw_documents_with_categories.items():

            if fixed_vocab:
                vocabulary = self.vocabularies_with_categories_[raw_category]
            else:
                # Add a new value when a new vocabulary item is seen
                vocabulary = defaultdict()
                vocabulary.default_factory = vocabulary.__len__

            analyze = self.build_analyzer()
            j_indices = []
            indptr = _make_int_array()
            values = _make_int_array()
            indptr.append(0)


            for doc in raw_documents:
                #category_dict[raw_category] = dict() #feature_counter
                feature_counter = collections.OrderedDict()
                for feature in analyze(doc):
                    try:
                        feature_idx = vocabulary[feature]
                        if feature_idx not in feature_counter:
                            feature_counter[feature_idx] = 1
                        else:
                            feature_counter[feature_idx] += 1
                    except KeyError:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue

                j_indices.extend(feature_counter.keys())
                values.extend(feature_counter.values())
                indptr.append(len(j_indices))

            if not fixed_vocab:
                # disable defaultdict behaviour
                vocabulary = dict(vocabulary)
                if not vocabulary:
                    raise ValueError("empty vocabulary; perhaps the documents only"
                                     " contain stop words")

            j_indices = np.asarray(j_indices, dtype=np.intc)
            indptr = np.frombuffer(indptr, dtype=np.intc)
            values = np.frombuffer(values, dtype=np.intc)

            X = sp.csr_matrix((values, j_indices, indptr),
                              shape=(len(indptr) - 1, len(vocabulary)),
                              dtype=self.dtype)
            X.sort_indices()
            _transferObject = TransferObjects(vocabulary, X)
            category_voc_dict[raw_category] = _transferObject
        return category_voc_dict

    def fit(self, raw_documents_with_categories, y=None):

        self.fit_transform(raw_documents_with_categories)
        return self

    def fit_transform(self, raw_documents_with_categories, y=None):

        #TODO: Labeled as to may be removed
        if isinstance(raw_documents_with_categories, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        _dict_for_transferObjects = self._count_vocab(raw_documents_with_categories,
                                          self.fixed_vocabulary_)

        self.dict_for_term_frequencies = collections.OrderedDict()
        self.dict_for_number_of_documents = collections.OrderedDict()
        self.number_of_categories = len(raw_documents_with_categories) - 1
        for raw_category, _transferObject in _dict_for_transferObjects.items():
            vocabulary = _transferObject.object1
            X = _transferObject.object2

            if self.binary:
                X.data.fill(1)

            if not self.fixed_vocabulary_:
                X = self._sort_features(X, vocabulary)

                n_doc = X.shape[0]
                max_doc_count = (max_df
                                 if isinstance(max_df, numbers.Integral)
                                 else max_df * n_doc)
                min_doc_count = (min_df
                                 if isinstance(min_df, numbers.Integral)
                                 else min_df * n_doc)
                if max_doc_count < min_doc_count:
                    raise ValueError(
                        "max_df corresponds to < documents than min_df")
                X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                           max_doc_count,
                                                           min_doc_count,
                                                           max_features)

                #'her bir kategorinin sozlugu'
                self.vocabularies_with_categories_[raw_category] = vocabulary

            #her bir categorinin sozlugunun, o kategoridaki dokumanlara oturtulmus hali ve bu sozcuklerin sayisi
            self.dict_for_term_frequencies[raw_category] = X

            '''
            #_of_Accounts_in_Current(Local)_Procedure
            '''
            # her bir kategoride kac dokouman var
            number_of_documents = X.shape[0]
            self.dict_for_number_of_documents[raw_category] = number_of_documents



        '''
        1 - #_of_Accounts_in_Current(Local)_Contain_This_Term
        2 - #_of_Accounts_in_Other(Global)_Contain_This_Term
        3 - #_of_Other_Procedures(Global)_Contain_This_Term
        '''
        self.dict_for_local_category_document_term_count = collections.OrderedDict()
        self.dict_for_global_category_document_term_count = collections.OrderedDict()
        self.dict_for_global_category_count = collections.OrderedDict()

        print('CountVectorizer for corpus are getting calculated')

        for _cat, _voc in self.vocabularies_with_categories_.items():
            print('CATEGORY: {}'.format(_cat))
            _sub_dict_for_local_category_document_term_count = collections.OrderedDict()
            _sub_dict_for_global_category_document_term_count = collections.OrderedDict()
            _sub_dict_for_global_category_count = collections.OrderedDict()

            corpus_size_of_category = len(_voc)
            global_index = 0
            old_percent = 0
            for _feature, _indx in _voc.items():
                # 1 - #_of_Accounts_in_Current(Local)_Contain_This_Term
                _arrs = self.dict_for_term_frequencies[_cat].toarray()[:, _indx]
                _sub_dict_for_local_category_document_term_count[_feature] = np.count_nonzero(_arrs)

                # 2 - 3 called_feature_globally
                _sub_dict_for_global_category_count, _sub_dict_for_global_category_count = self._count_feature_globally(_cat, _feature, self.dict_for_term_frequencies, _sub_dict_for_global_category_document_term_count, _sub_dict_for_global_category_count)

                self.dict_for_global_category_document_term_count[_cat] = _sub_dict_for_global_category_count

                self.dict_for_global_category_count[_cat] = _sub_dict_for_global_category_count

                percent = round(((global_index + 1) / corpus_size_of_category) * 100)
                if old_percent != percent:
                    print("{} % of data processed for {} category".format(str(percent), _cat))
                    old_percent = percent
                global_index += 1

            self.dict_for_local_category_document_term_count[_cat] = _sub_dict_for_local_category_document_term_count


        return self.vocabularies_with_categories_, self.number_of_categories, self.dict_for_term_frequencies,self.dict_for_number_of_documents, \
               self.dict_for_local_category_document_term_count,self.dict_for_global_category_document_term_count, self.dict_for_global_category_count

    def _count_feature_globally(self, excluded_category, feature, term_freq_data_with_categories, _sub_dict_for_global_category_document_term_count, _sub_dict_for_global_category_count ):
        '''
        2 - #_of_Accounts_in_Other(Global)_Contain_This_Term
        3 - #_of_Other_Procedures(Global)_Contain_This_Term
        '''

        for _cat, _voc in self.vocabularies_with_categories_.items(): #business - gs
            if _cat not in excluded_category: # sport

                try:
                    _indx = _voc[feature] # business daki sozluk ten gs indexi cekiliyor. # olmaayabilir boyle bir sozcuk
                    _arrs = term_freq_data_with_categories[_cat].toarray()[:, _indx] # eger boyle bir sozcuk varsa onlarin sayilari aliniyor business categorisinde
                    document_incrementer =  np.count_nonzero(_arrs)
                    category_incrementer = 1
                except:
                    document_incrementer = 0
                    category_incrementer = 0
                finally:
                    if feature not in _sub_dict_for_global_category_document_term_count: # eger daha onceden boyle bir sozcuk eklenmemisse ilk cikan rakam,
                        _sub_dict_for_global_category_document_term_count[feature] = document_incrementer
                        _sub_dict_for_global_category_count[feature] = category_incrementer
                    else:
                        _sub_dict_for_global_category_document_term_count[feature] += document_incrementer
                        _sub_dict_for_global_category_count[feature] += category_incrementer

        return _sub_dict_for_global_category_document_term_count, _sub_dict_for_global_category_count

    def transform(self, raw_documents_with_categories):

        # TODO: Labeled as to may be removed
        if isinstance(raw_documents_with_categories, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        if not hasattr(self, 'vocabularies_with_categories_'): #means, there is a vocabulary_ attr of self or not?
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
       # _, X = self._count_vocab(raw_documents_with_categories, fixed_vocab=True)

        _dict_for_transferObjects = self._count_vocab(raw_documents_with_categories, fixed_vocab=True)

        dict_for_transforms = collections.OrderedDict()
        for raw_category, _transferObject in _dict_for_transferObjects.items():
            _ = _transferObject.object1
            X = _transferObject.object2
            if self.binary:
                X.data.fill(1)
            dict_for_transforms[raw_category] = X
        return dict_for_transforms

    def inverse_transform(self, X):

        self._check_vocabulary()

        if sp.issparse(X):
            # We need CSR format for fast row manipulations.
            X = X.tocsr()
        else:
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        self._check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]


def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class TfidfTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):

        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self

    def transform(self, X, copy=True):

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

class DfIcfTransformer(object):
    def __init__(self, vocabularies_with_categories,
                 number_of_categories,
                 dict_for_term_frequencies,
                 dict_for_number_of_documents,
                 dict_for_local_category_document_term_count,
                 dict_for_global_category_document_term_count,
                 dict_for_global_category_count,
                 norm='l2', smooth=True):
        self.norm = norm
        self.use_idf = True
        self.smooth_idf = smooth
        self.sublinear_tf = False
        self.vocabularies_with_categories = vocabularies_with_categories
        self.number_of_categories = number_of_categories
        self.dict_for_term_frequencies = dict_for_term_frequencies
        self.dict_for_number_of_documents = dict_for_number_of_documents
        self.dict_for_local_category_document_term_count = dict_for_local_category_document_term_count
        self.dict_for_global_category_document_term_count = dict_for_global_category_document_term_count
        self.dict_for_global_category_count = dict_for_global_category_count
        self.dficf = collections.OrderedDict()

    def __avoid_zero_division(self, val):
        _init_val = 0.1  # 0.000000000001#
        if val <= 0:
            val = _init_val
        return val

    def get_global_vectors(self):

        #global dictionary is getting filed
        self.global_vocabulary=collections.OrderedDict()
        global_vec_pntr = 0
        for _cat, _dict in self.vocabulary_for_each_category_:
            for _voc in _dict.items():
                if _voc not in self.global_vocabulary:
                    self.global_vocabulary[_voc] = global_vec_pntr
                    global_vec_pntr += 1

        from sklearn.utils import Bunch
        '''
        #Container object for datasets
        return Bunch(data=data, target=target,
                         target_names=target_names,
                         DESCR=fdescr,
                         feature_names=['sepal length (cm)', 'sepal width (cm)',
                                        'petal length (cm)', 'petal width (cm)'])

        '''

        for _subkey, _subval in self.dficf_weighs_for_terms_in_each_category_[_cat].items():

            print('Term ({}) : dficf weight ({})'.format(_subkey, _subval))



    def transform(self):

        print('DFICF weights are getting calculated')

        self.dict_for_term_dcicf_weights_for_each_category = collections.OrderedDict()
        for _cat, _docs in self.dict_for_term_frequencies.items():
            print('CATEGORY: {}'.format(_cat))
            _docs = _docs.toarray()
            _num_of_docs_in_current_category = self.dict_for_number_of_documents[_cat]
            _num_of_docs_in_other_categories = sum(self.dict_for_number_of_documents.values()) -  _num_of_docs_in_current_category
            _num_of_other_categories = self.number_of_categories
            self.dict_for_term_dcicf_weights_for_each_category[_cat]=collections.OrderedDict()

            j_indices = []
            indptr = _make_int_array()
            values =  array.array(str('f'))
            indptr.append(0)
            corpus_size_of_category = len(_docs)
            global_index = 0
            old_percent = 0

            for _doc_indx, _doc in enumerate(_docs):
                _total_features_in_document = sum(_doc)
                doc_dficf_weigts = collections.OrderedDict()
                for _feature, _indx in  self.vocabularies_with_categories[_cat].items():

                    _term = _feature

                    # -----------
                    # TF
                    # -----------
                    _term_freq = _doc[_indx]
                    _tf = _term_freq / _total_features_in_document  # term_frequency

                    # -----------
                    # DF
                    # -----------
                    _num_of_docs_in_current_category_contain_this_term = self.dict_for_local_category_document_term_count[_cat][_term]
                    _df = _num_of_docs_in_current_category_contain_this_term / _num_of_docs_in_current_category  # document_frequency

                    # -----------
                    # doc-icf
                    # -----------
                    _num_of_docs_in_other_categories_contain_this_term = self.dict_for_global_category_document_term_count[_cat][_term]
                    _num_of_docs_in_other_categories_contain_this_term = self.__avoid_zero_division(_num_of_docs_in_other_categories_contain_this_term)
                    _doc_icf = math.log(_num_of_docs_in_other_categories / _num_of_docs_in_other_categories_contain_this_term, 10)
                    _doc_icf = self.__avoid_zero_division(_doc_icf)

                    # -----------
                    # cat-icf
                    # -----------
                    _num_of_categories_contain_this_term = self.dict_for_global_category_count[_cat][_term]
                    _num_of_categories_contain_this_term = self.__avoid_zero_division(
                        _num_of_categories_contain_this_term)
                    _cat_icf = math.log(_num_of_other_categories / _num_of_categories_contain_this_term, 10)
                    _cat_icf = self.__avoid_zero_division(_cat_icf)

                    # -----------
                    # GOD : TF x DF x IDF x IGF
                    # -----------
                    _tf_df_doc_icf_cat_icf = _tf * _df* _doc_icf * _cat_icf

                    # -----------
                    # GOD : DF x IDF x IGF
                    # -----------
                    _df_doc_icf_cat_icf = _df * _doc_icf * _cat_icf

                    doc_dficf_weigts[_indx] = _tf_df_doc_icf_cat_icf
                    self.dict_for_term_dcicf_weights_for_each_category[_cat][_term] = _df_doc_icf_cat_icf

                j_indices.extend(doc_dficf_weigts.keys())
                values.extend(doc_dficf_weigts.values())
                indptr.append(len(j_indices))

                percent = round(((global_index + 1) / corpus_size_of_category) * 100)
                if old_percent != percent:
                    print("{} % of data processed for {} category".format(str(percent), _cat))
                    old_percent = percent
                global_index += 1

            # j_indices = np.asarray(j_indices, dtype=np.intc)
            # indptr = np.frombuffer(indptr, dtype=np.intc)
            # values = np.frombuffer(values, dtype=np.intc)

            X = sp.csr_matrix((values, j_indices, indptr),
                              shape=(len(indptr) - 1, len(self.vocabularies_with_categories[_cat])),
                              dtype=np.float64)
            X.sort_indices()
            self.dficf[_cat] = X

        self.__merge_vocabularies()
        return self


    @property
    def dficf_weighs_for_documents_in_each_category_(self):

        #return np.ravel(self._idf_diag.sum(axis=0))
        return self.dficf

    @property
    def vocabulary_for_each_category_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self.vocabularies_with_categories

    def reversed_vocabulary_for_each_category(self):
        try:
            checker = self.reversed_dict_for_term_dcicf_weights_for_each_category
        except:
            self.reversed_dict_for_term_dcicf_weights_for_each_category = collections.OrderedDict()
            self.sorted_reversed_dict_for_term_dcicf_weights_for_each_category = collections.OrderedDict()
            for _cat, _val in self.dficf_weighs_for_documents_in_each_category_.items():
                _voc = self.vocabulary_for_each_category_[_cat]
                _voc_reversed = {y: x for x, y in _voc.items()}
                _sorted_reversed_voc = sorted(_voc.items(), key=operator.itemgetter(0))
                self.reversed_dict_for_term_dcicf_weights_for_each_category[_cat] = _voc_reversed
                self.sorted_reversed_dict_for_term_dcicf_weights_for_each_category[_cat] = _sorted_reversed_voc

        return self.reversed_dict_for_term_dcicf_weights_for_each_category, self.sorted_reversed_dict_for_term_dcicf_weights_for_each_category

    @property
    def reversed_vocabulary_for_each_category_(self):

        me, other = self.reversed_vocabulary_for_each_category()
        return me

    @property
    def reversed_ordered_vocabulary_for_each_category_(self):

        other, me = self.reversed_vocabulary_for_each_category()
        return me

    @property
    def dficf_weighs_for_terms_in_each_category_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self.dict_for_term_dcicf_weights_for_each_category

    def _get_term_dic_for_each_document(self, vocabulary, document_list):

        # in order to access the each documents term - freq values, i kept them in a dict.
        from collections import defaultdict
        document_term_list = []
        for document in document_list:
            d = collections.OrderedDict()
            for _term, _index in vocabulary.items():
                d[_term] = document[_index]
            document_term_list.append(d)
        return document_term_list

    def fit_transform(self, data, _stemming_and_lemmatization=True, is_random=False):

        from .copy_of_orj_tfidf_vectorizer import CountVectorizer as org_count_vectorizer
        count_vectorizer = org_count_vectorizer(stop_words='english', stemming_and_lemmatization=_stemming_and_lemmatization)
        count_vectorizer.fit_transform(data)
        self.new_doc_voc = count_vectorizer.vocabulary_
        #new_doc_voc = collections.OrderedDict(sorted(new_doc_voc.items()))
        new_doc_list = count_vectorizer.transform(data).toarray()

        self.term_dic_for_each_document = self._get_term_dic_for_each_document(self.new_doc_voc, new_doc_list)

        self.transformed_input_data_for_each_category_voc = collections.OrderedDict()
        print('New Data is getting fitted and transformed')
        for _cat, _vocs in self.vocabularies_with_categories.items():
            print('CATEGORY: {}'.format(_cat))
            _vocs = collections.OrderedDict(sorted(_vocs.items()))
            _new_docs = []
            for _docIndx in range(len(new_doc_list)):
                _doc_feature_dict = self.term_dic_for_each_document[_docIndx]
                _new_doc_elements = []
                for _cat_term, _cat_term_indx in _vocs.items():
                    try:
                        _doc_term_freq = _doc_feature_dict[_cat_term]
                    except:
                        _doc_term_freq = 0
                    finally:
                        try:
                            _dficf_weight = _doc_term_freq * self.dficf_weighs_for_terms_in_each_category_[_cat][_cat_term]
                        except:
                            _dficf_weight = 0
                        finally:
                            #term_indx = new_doc_voc[_cat_term]
                            _new_doc_elements.append(_dficf_weight)
                _new_docs.append(_new_doc_elements)
            self.transformed_input_data_for_each_category_voc[_cat] = _new_docs

        self._create_vector(is_random)
        self._calculate_similarity_score()
        self._calculate_euclidean_score()
        self._calculate_cosine_score()
        return self


    def _create_vector(self, is_random=False):
        X_y_y_index = []
        self.category_as_dict_ = collections.OrderedDict()
        for _i,_elements in enumerate(self.transformed_input_data_for_each_category_voc.items()):
            _cat = _elements[0]
            _sentences = _elements[1]
            for _si, _sentence in enumerate(_sentences):
                x = [0.0] * len(self.merged_vocabularies_)
                for _wi, _categorical_value_of_word in enumerate(_sentence):
                    _word_name = self.reversed_vocabulary_for_each_category_[_cat][_wi]
                    _index_of_word_in_merged_voc = self.merged_vocabularies_[_word_name]
                    x[_index_of_word_in_merged_voc] = _categorical_value_of_word
                X_y_y_index.append((x, _i, _cat, _si))
            self.category_as_dict_[_i] = _cat

        if is_random is True:
            random.shuffle(X_y_y_index)

        self.X = np.array(X_y_y_index)[:,0]
        self.y = np.array(X_y_y_index)[:,1]
        self.y_ = np.array(X_y_y_index)[:,2]
        self.index = np.array(X_y_y_index)[:,3]
        self.all_together = X_y_y_index
        return self.X, self.y, self.y_, self.index, self.all_together

    @property
    def vectors_(self):
        return self.X, self.y, self.y_, self.index, self.all_together

    @property
    def categories_(self):
        return self.category_as_dict_


    def sort_voc(self, dict):
        return sorted(dict.items(), key=operator.itemgetter(0))

    def reverse_voc(self, dict):
        return {y: x for x, y in dict.items()}


    def _calculate_euclidean_distance(self, v1, v2):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))

    def _calculate_euclidean_score(self):
        self._similarity_euclidean_distance = []
        for _doc_indx in range(len(self.term_dic_for_each_document)):
            final_new_document_distance_score = 1000000000000000
            score_details = []
            for _cat in self.transformed_input_data_for_each_category_voc_.keys():
                _new_transformed_document = self.transformed_input_data_for_each_category_voc_[_cat][_doc_indx]
                _trained_document = self.dficf_weighs_for_documents_in_each_category_[_cat][0].toarray()[0]
                _new_document_distance_score = spatial.distance.euclidean(_new_transformed_document, _trained_document)
                score_detail=collections.OrderedDict()
                # if _new_document_distance_score <= final_new_document_distance_score:
                #     final_new_document_distance_score = _new_document_distance_score
                score_detail['score'] = _new_document_distance_score
                score_detail['category'] = _cat
                score_details.append(score_detail)
            self._similarity_euclidean_distance.append(score_details)
        return self

    @property
    def similarity_euclidean_distance_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self._similarity_euclidean_distance

    def _calculate_cosine_score(self):
        self._calculate_cosine_distance = []
        for _doc_indx in range(len(self.term_dic_for_each_document)):
            final_new_document_distance_score = 1000000000000000
            score_details = []
            for _cat in self.transformed_input_data_for_each_category_voc_.keys():
                _new_transformed_document = self.transformed_input_data_for_each_category_voc_[_cat][_doc_indx]
                _trained_document = self.dficf_weighs_for_documents_in_each_category_[_cat][0].toarray()[0]
                _new_document_distance_score =  1 - spatial.distance.cosine(_new_transformed_document, _trained_document)
                score_detail = collections.OrderedDict()
                # if _new_document_distance_score <= final_new_document_distance_score:
                #     final_new_document_distance_score = _new_document_distance_score
                score_detail['score'] = (0 if math.isnan(_new_document_distance_score) is True else _new_document_distance_score)
                score_detail['category'] = _cat
                score_details.append(score_detail)
            self._calculate_cosine_distance.append(score_details)
        return self

    @property
    def similarity_cosine_distance_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self._calculate_cosine_distance

    def _calculate_similarity_score(self):
        self._similarity_scores = []
        for _new_document in self.term_dic_for_each_document:
            final_new_document_similarity_score = 0
            score_details = []
            for _cat, _cat_weights in self.dficf_weighs_for_terms_in_each_category_.items():
                _term_weight_for_new_document_in_cat= 0.00000001
                _cat_based_term_list=[]
                for _term, _temp_value in _new_document.items():
                    try:
                        _term_weight_for_new_document_in_cat += self.dficf_weighs_for_terms_in_each_category_[_cat][_term]
                        _cat_based_term_list.append(_term)
                    except:
                        _term_weight_for_new_document_in_cat += 0

                total_term_weight_for_new_document_in_all_categories = self._get_total_weights_in_all_categories(_cat_based_term_list)
                _new_document_similarity_score = _term_weight_for_new_document_in_cat / total_term_weight_for_new_document_in_all_categories
                score_detail = collections.OrderedDict()
                # if _new_document_similarity_score >= final_new_document_similarity_score:
                #     final_new_document_similarity_score = _new_document_similarity_score
                score_detail['score'] = _new_document_similarity_score
                score_detail['category'] = _cat
                score_details.append(score_detail)
            self._similarity_scores.append(score_details)
        return self

    @property
    def similarity_scores_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self._similarity_scores


    def _get_total_weights_in_all_categories(self, cat_based_term_list):
        total_term_weight_for_new_document_in_all_categories =  0.00000001
        for _cat, _cat_weights in self.dficf_weighs_for_terms_in_each_category_.items():
            for _term in cat_based_term_list:
                try:
                    total_term_weight_for_new_document_in_all_categories += self.dficf_weighs_for_terms_in_each_category_[_cat][_term]
                except:
                    total_term_weight_for_new_document_in_all_categories += 0
        return total_term_weight_for_new_document_in_all_categories

    @property
    def transformed_input_data_for_each_category_voc_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self.transformed_input_data_for_each_category_voc

    @property
    def new_doc_voc_(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self.new_doc_voc

    def __merge_vocabularies(self):
        try:
            checker = self.merged_vocabularies_
        except:
            self.merged_vocabularies_= collections.OrderedDict()

            for _cat, _val in self.dficf_weighs_for_documents_in_each_category_.items():
                self.merged_vocabularies_ = {**self.merged_vocabularies_, **self.vocabulary_for_each_category_[_cat]}

            #self.merged_vocabularies_ = dict(enumerate(_key for _key, _val in self.merged_vocabularies_.items()))
            self.merged_vocabularies_ = {x[0]: i for i, x in enumerate(self.merged_vocabularies_.items())}

        return self.merged_vocabularies_

    @property
    def merged_vocabularies(self):

        # return np.ravel(self._idf_diag.sum(axis=0))
        return self.merged_vocabularies_


class TfidfVectorizer(CountVectorizer):


    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    def fit(self, raw_documents, y=None):

        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):

        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):

        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(TfidfVectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

class TransferObjects(object):
    def __init__(self, object1, object2, object3=None):
        self._object1 = object1
        self._object2 = object2
        self._object3 = object3


    @property
    def object1(self):
        return self._object1

    @object1.setter
    def object1(self, value):
        self._object1 = value

    @property
    def object2(self):
        return self._object2

    @object2.setter
    def object2(self, value):
        self._object2 = value

    @property
    def object3(self):
        return self._object3

    @object3.setter
    def object3(self, value):
        self._object3 = value