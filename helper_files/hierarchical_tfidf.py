import math
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np


class TFIDF(object):
    def __init__(self,
                 proc_data,
                 current_proc_total_accounts,
                 current_proc_doc_count_dict,
                 current_proc_term_count_dict,
                 other_procs_total_accounts,
                 other_procs_doc_count_dict,
                 other_procs,
                 other_procs_count_dict,
                 ):
        self.__current_proc_data = proc_data

        self.__current_proc_total_accounts = current_proc_total_accounts
        self.__current_proc_doc_count_dict = current_proc_doc_count_dict
        self.__current_proc_term_count_dict = current_proc_term_count_dict

        self.__other_procs_total_accounts = other_procs_total_accounts
        self.__other_procs_doc_count_dict = other_procs_doc_count_dict

        self.__other_procs = other_procs
        self.__other_procs_count_dict = other_procs_count_dict

        self.result_colum_names = ['Procedure', 'AccountID', '#_of_Terms_in_Account',
                                   'Term',
                                   'Term_Freq_in_Account',
                                   '#_of_Accounts_in_Current(Local)_Procedure',
                                   '#_of_Accounts_in_Current(Local)_Contain_This_Term',
                                   '#_of_Accounts_in_Other(Global)_Procedures',
                                   '#_of_Accounts_in_Other(Global)_Contain_This_Term',
                                   '#_of_Other_Procedures(Global)',
                                   '#_of_Other_Procedures(Global)_Contain_This_Term',
                                   'Current(Local)_TF',
                                   'Current(Local)_DF',
                                   'Other_Docs_(Global)_IDF',
                                   'Other_Procedures_(Global)_IGF',
                                   'GOD_TFxDFxIDFxIGF',
                                   'GOD_DFxIDFxIGF']
        self.result_as_dataframe = None
        self.result_as_arr = []

    def __avoid_zero_division(self, val):
        _init_val = 0.1  # 0.000000000001#
        if val <= 0:
            val = _init_val
        return val

    def execute(self):
        self.__calculate()

    def __calculate(self):

        for index, row in self.__current_proc_data.iterrows():
            _procedure = row['Procedure']
            _accountid = row['AccountID']
            _features = [document for document in row['Features'].split()]
            _total_features = len(_features)
            _count_vectorizer = CountVectorizer()
            _term_freq_matrix = _count_vectorizer.fit_transform(
                list([row['Features']]))
            _dictionary = _count_vectorizer.vocabulary_
            # get voc index of matrix
            for key, value in _dictionary.items():
                _term = key
                _index_of_item_in_matrix = np.where(_term_freq_matrix.indices == value)[0][0]

                _term_freq = _term_freq_matrix.data[_index_of_item_in_matrix]
                # -----------
                # CURRENT(LOCAL) PART
                # -----------
                _local_tf = _term_freq / _total_features #term_frequency
                _num_of_Accounts_in_Current_Procedure = self.__current_proc_total_accounts
                _num_of_Accounts_in_Current_Contain_This_Term = self.__current_proc_doc_count_dict[_term]
                _local_df = _num_of_Accounts_in_Current_Contain_This_Term / _num_of_Accounts_in_Current_Procedure #document_frequency

                # -----------
                # OTHERS(GLOBAL) DOC PART
                # -----------
                _num_of_Accounts_in_Other_Procedures = self.__other_procs_total_accounts
                _num_of_Accounts_in_Other_Contain_This_Term = self.__other_procs_doc_count_dict[_term]
                _num_of_Accounts_in_Other_Contain_This_Term = self.__avoid_zero_division(_num_of_Accounts_in_Other_Contain_This_Term)
                _other_doc_idf = math.log(_num_of_Accounts_in_Other_Procedures / _num_of_Accounts_in_Other_Contain_This_Term,10)
                _other_doc_idf = self.__avoid_zero_division(_other_doc_idf)

                # -----------
                # OTHERS(GLOBAL) PROC/CATEGORY/GROUP PART
                # -----------
                _num_of_Other_Procedures = self.__other_procs
                _num_of_Other_Procedures_Contain_This_Term = self.__other_procs_count_dict[_term] - 1
                _num_of_Other_Procedures_Contain_This_Term = self.__avoid_zero_division(_num_of_Other_Procedures_Contain_This_Term)
                _other_group_igf = math.log(_num_of_Other_Procedures / _num_of_Other_Procedures_Contain_This_Term, 10)
                _other_group_igf = self.__avoid_zero_division(_other_group_igf)

                # -----------
                # GOD : TF x DF x IDF x IGF
                # -----------
                _god_tf_df_idf_igf = _local_tf * _local_df * _other_doc_idf * _other_group_igf

                # -----------
                # GOD : DF x IDF x IGF
                # -----------
                _god_df_idf_igf = _local_df * _other_doc_idf * _other_group_igf

                self.result_as_arr.append((_procedure, _accountid, _total_features, _term, _term_freq,
                                           _num_of_Accounts_in_Current_Procedure,
                                           _num_of_Accounts_in_Current_Contain_This_Term,
                                           _num_of_Accounts_in_Other_Procedures,
                                           _num_of_Accounts_in_Other_Contain_This_Term,
                                           _num_of_Other_Procedures,
                                           _num_of_Other_Procedures_Contain_This_Term,
                                           _local_tf,
                                           _local_df,
                                           _other_doc_idf,
                                           _other_group_igf,
                                           _god_tf_df_idf_igf,
                                           _god_df_idf_igf))

        self.result_as_dataframe = pd.DataFrame(data=self.result_as_arr, columns=self.result_colum_names)
