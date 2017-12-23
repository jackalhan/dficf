import os
import datetime as dt
import math
import string

from sklearn.model_selection import train_test_split

from innovations.dficf import hierarchical_tfidf
from sklearn.datasets import load_iris
import collections
from itertools import chain
import codecs
import pandas as pd
# from nltk.corpus import stopwords
import re
from stop_words import get_stop_words
from scipy import spatial
import pickle
import operator
import innovations.utils as util
from sklearn.utils import Bunch
import numpy as np
import random

'''
#Container object for datasets
return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])

'''
FOLDER_SEPERATOR = "/"
__GENERATED_DATA_FOLDER = 'generated_data'
__FOLDER_NAME = 'bbc'
__BASE_FOLDER = r'/home/jackalhan/Development/app_data/tf_idf_hiarchical'
TARGET_PATH = os.path.join(__BASE_FOLDER, __FOLDER_NAME)
tokenize = lambda doc: doc.lower().split(" ")


def get_files_content(root_folder, file_extension):
    root_folder_name = root_folder.rsplit(FOLDER_SEPERATOR, 1)[1]
    file_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("." + file_extension):
                file_path = os.path.join(root, file)
                parent_folder = root.rsplit(FOLDER_SEPERATOR, 1)[1]
                file_name = file.split('.')[0]
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as content_file:
                        content = content_file.read()
                        exclude = set(string.punctuation)
                        content = ''.join(ch.replace("\n", "-") for ch in content if ch not in exclude)
                        content = content.replace("--", " ")
                        content = content.replace("-", " ").strip()
                        # NLTK - STOPWORDS
                        # content = " ".join(
                        #     word.strip() for word in content.split() if word not in stopwords.words('english'))
                        # OPEN SOURCE GITHUB - STOP_WORDS
                        content = " ".join(
                            word.strip() for word in content.split() if word not in get_stop_words('english'))
                        content = re.sub(r'\d+', '', content)

                    file_list.append((parent_folder, file_name, content.lower()))
                except Exception as e:
                    print(str(e))
                    print(parent_folder, file)
    return root_folder_name, pd.DataFrame(data=file_list,
                                          columns=['Procedure', 'AccountID', 'Features'])


def hiarchical_dficf(data_size_rate, is_test_run=False, save=False):
    from innovations.dficf.modified_dficf_vectorizer import CountVectorizer, TfidfTransformer, DfIcfTransformer

    # NEW WAY TO DO THAT
    if is_test_run is False:
        start_time = dt.datetime.now()
        print('Files are getting ready')
        root_folder_name, files_dataframe = get_files_content(TARGET_PATH, 'txt')
        print('{} of files are parsed in {} mins.'.format(str(len(files_dataframe)),
                                                          ((dt.datetime.now() - start_time).seconds) / 60))
        np.random.seed(0)
        data = files_dataframe.groupby(['Procedure'])
        len_of_data = len(data)
        train_set = collections.OrderedDict()
        test_set = collections.OrderedDict()
        for _proc, _grouped in data:
            doc_size = len(_grouped.Features)
            print("Data size of {} proc is {}".format(_proc, doc_size))
            print("With adding an data rate {}".format(int(doc_size / data_size_rate)))
            docs = _grouped.Features.transform(np.random.permutation)
            features_train, features_test = train_test_split(docs[:int(doc_size / data_size_rate)], test_size=0.33)
            train_set[_proc] = list()
            # train_set[_proc].extend([x for x in _grouped.Features])
            train_set[_proc].extend([x for x in features_train])
            test_set[_proc] = list()
            test_set[_proc].extend([x for x in features_test])

    else:
        train_set = collections.OrderedDict()
        # test_set = dict()

        train_set['sport'] = [
            # "Katerina Thanou is confident she and fellow sprinter Kostas Kenteris will not be punished for missing drugs tests before the Athens Olympics."
            # "The Greek pair appeared at a hearing on Saturday which will determine whether their provisional bans from athletics ruling body the IAAF should stand. "
            # "After five months we finally had the chance to give explanations. I am confident and optimistic, said Thanou. The athletes lawyer Grigoris Ioanidis "
            # "said he believed the independent disciplinary committee set up by the Greek Athletics Federation would find them innocent. We are almost certain that "
            # "the charges will be dropped, said Ioanidis.",
            #
            # "London Marathon organisers are hoping that banned athlete Susan Chepkemei will still take part in this year's race on 17 April.Chepkemei was suspended from all competition "
            # "until the end of the year by Athletics Kenya after failing to report to a national training camp. We are watching it closely said London race director David Bedford. The camp "
            # "in Embu was to prepare for the IAAF World Cross Country Championships later this month. Chepkemei however took part and finished third in last Sunday's world best 10K race in "
            # "Puerto Rico.",
            #
            # "Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross-Country Championships. The 31-year-old is concerned the event, which "
            # "starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. Radcliffe was world cross-country champion in 2001 and 2002 but missed "
            # "last year's event because of injury. In her absence, the GB team won bronze in Brussels.",
            #
            # "Wayne Rooney made a winning return to Everton as Manchester United cruised into the FA Cup quarter-finals."
            # "Rooney received a hostile reception, but goals in each half from Quinton Fortune and Cristiano Ronaldo silenced the jeers at Goodison Park. Fortune headed home after "
            # "23 minutes before Ronaldo scored when Nigel Martyn parried Paul Scholes' free-kick. Marcus Bent missed Everton's best chance when Roy Carroll, who was later struck by "
            # "a missile, saved at his feet. It was a fantastic performance by us. In fairness I think Everton have missed a couple of players and got some young players out.",
            #
            # "A brave defensive display, led by keeper David James, helped Manchester City hold the leaders Chelsea. After a quiet opening, James denied Damien Duff, Jiri Jarosik and "
            # "Mateja Kezman, while Paul Bosvelt cleared William Gallas' header off the line. Robbie Fowler should have scored for the visitors but sent his header wide. Chelsea had "
            # "most of the possession in the second half but James kept out Frank Lampard's free-kick and superbly tipped the same player's volley wide. City went into the game with the "
            # "proud record of being the only domestic team to beat Chelsea this season."
            "football ball player", "ball player"
        ]

        train_set['tech'] = [
            # "Before now many spammers have recruited home PCs to act as anonymous e-mail relays in an attempt to hide the origins of their junk mail. The PCs are recruited using viruses "
            # "and worms that compromise machines via known vulnerabilities or by tricking people into opening an attachment infected with the malicious program. Once compromised the machines "
            # "start to pump out junk mail on behalf of spammers. Spamhaus helps to block junk messages from these machines by collecting and circulating blacklists of net addresses known to "
            # "harbour infected machines. But the novel worm spotted recently by Spamhaus routes junk via the mail servers of the net service firm that infected machines used to get online in "
            # "the first place. In this way the junk mail gets a net address that looks legitimate. As blocking all mail from net firms just to catch the spam is impractical, Spamhaus is worried "
            # "that the technique will give junk mailers the ability to spam with little fear of being spotted and stopped.",
            #
            # "Use of games in schools has been patchy she found, with Sim City proving the most popular. Traditionally schools have eschewed mainstream games in favour of used so-called edu-tainment "
            # "software in a belief that such packages help to make learning fun, she found in her research. It is perhaps in a compromise between edutainment and mainstream games that the greatest "
            # "potential for classroom useable games lies, she wrote in a paper entitled Games and Learning. 'Lite' versions of existing games could be the way forward and would overcome one of the "
            # "biggest hurdles - persuading developers to write for the educational market. This would appeal to developers because of the low costs involved in adapting them as well as offering a "
            # "new opportunity for marketing. Already there are games on the market, such as Civilisation and Age of Empire, that have educational elements said Mr Owen. Even in Grand Theft Auto it "
            # "is not just the violence that engages people, he said. It could be some time until that particular game makes it into the classroom though.",
            #
            # "Microsoft's Media Player 10 took the award for Most Wanted Software. This year was the 10th anniversary of the PC Pro awards, which splits its prizes into two sections. The first are "
            # "chosen by the magazine's writers and consultants, the second are voted for by readers. Mr Trotter said more than 13,000 people voted for the Reliability and Service Awards, "
            # "twice as many as in 2003. Net-based memory and video card shop Crucial shared the award for Online Vendor of the year with Novatech."
            "football game console", "game console"
        ]

        train_set['entertainment'] = [
            # "If the ratings are confirmed, the episode will have given the soap its highest audience for a year. The overnight figures showed almost 60% of the viewing public tuned into EastEnders "
            # "between 2000 and 2100 GMT, leaving ITV1 with about 13%. We are very pleased with the figures,a BBC spokesman said. It shows viewers have really enjoyed the story of Den's demise. "
            # "The show's highest audience came at Christmas 1986, when more than 30 million tuned in to see Den, played by Leslie Grantham, hand divorce papers to wife Angie.",
            #
            # "Sky first scooped Oscar rights from the BBC in 1999, but the BBC won them back in 2001 when Sky was forced to pull out of a bidding war due to financial constraints. BBC viewers will "
            # "of course be able to watch quality coverage of the 2005 Academy Awards on the BBC's bulletins and news programmes, a spokesman said. Among the films tipped to do well at this year's "
            # "Academy Awards are Martin Scorsese's The Aviator, Jean-Pierre Jeunet's A Very Long Engagement and the Ray Charles biopic, Ray."
            "cinema console actor", "cinema actor"
        ]

        test_set = ['football season is getting started']

        print('train set:', train_set)
        print('test set:', test_set)

    count_vectorizer = CountVectorizer(stop_words='english', stemming_and_lemmatization=True)

    # categorical values from the trained count vectorizer.
    vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies, dict_for_number_of_documents, \
    dict_for_local_category_document_term_count, dict_for_global_category_document_term_count, \
    dict_for_global_category_count = count_vectorizer.fit_transform(train_set)

    del count_vectorizer

    dficf = DfIcfTransformer(vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies,
                             dict_for_number_of_documents,
                             dict_for_local_category_document_term_count, dict_for_global_category_document_term_count,
                             dict_for_global_category_count)

    dficf.transform()
    # print('Merged vocabularies:', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)))
    # for _cat, _val in dficf.dficf_weighs_for_documents_in_each_category_.items():
    #     print("-" * 50)
    #     print(_cat)
    #     print("-" * 50)
    #
    #     # print(10 * '*')
    #     # print("dficf_weighs_for_documents_in_each_category:", _val.toarray())
    #     #
    #     # print(10 * '*')
    #     # print("vocabulary_for_each_category")
    #     # for _subkey, _subval in dficf.vocabulary_for_each_category_[_cat].items():
    #     #     print('Term ({}) : Position/Index ({})'.format(_subkey, _subval))
    #     # print(10 * '*')
    #     # print("dcicf_weights_for_terms_in_each_category")
    #     # for _subkey, _subval in dficf.dficf_weighs_for_terms_in_each_category_[_cat].items():
    #     #     print('Term ({}) : dficf weight ({})'.format(_subkey, _subval))
    #     _voc = dficf.vocabulary_for_each_category_[_cat]
    #     _reversed = dficf.reversed_vocabulary_for_each_category_[_cat]
    #     _sorted_reversed = dficf.reversed_ordered_vocabulary_for_each_category_[_cat]
    #
    #     print('Normal Voc:', _voc)
    #     print('Reversed Voc:', _reversed)
    #     print('Sorted Reversed Voc:', _sorted_reversed)

    # Just add the known items to the test set from the test set.
    # Start
    X, y, X_y_df = extract_elements_from_2Ddict(train_set)
    X_y_df.to_csv(os.path.join(TARGET_PATH, 'train_documents.csv'))
    # end
    del train_set
    dficf.fit_transform(X)
    # print('test_set_voc_', dficf.new_doc_voc_)
    # print('test_set_terms', dficf.term_dic_for_each_document)
    # print('transformed_input_data_for_each_category_voc_', dficf.transformed_input_data_for_each_category_voc_)
    X_data, y_data, y_data_, sentence_index, all_together = dficf.vectors_


    # print('vectors of test set', dficf.vectors_)
    # print('categories as dict', dficf.category_as_dict_)

    # Just add the known items to the test set from the test set.
    # Start
    len_of_test = len(X)
    len_of_test_ = len(X_data)
    X_data_ = []
    y_data__ = []
    for _i, _y in enumerate(y):
        print('*' * 10)
        print('Document index:', str(_i))
        for _i_vectors in range(_i, len_of_test_, len_of_test):
            print('Checking index in Data:', str(_i_vectors))
            if (_y == y_data_[_i_vectors]):
                print('Found index in Data:', str(_i_vectors))
                print('Found doc in Data:', _y)
                X_data_.append(X_data[_i_vectors])
                y_data__.append(y_data_[_i_vectors])
                break

    # End
    all_together = list(zip(X_data_, y_data__))
    train_data_all_in_one = pd.DataFrame(data=all_together,
                                         columns=['vector', 'generated_category_name'])

    train_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'dficf_train_all_in_one.csv'))

    util.dump_adff('DFICF', 'DFICF BBC DATA', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)),
                   dficf.category_as_dict_, X_data_, y_data__, TARGET_PATH, 'dficf_train_bbc_arff')

    # Just add the known items to the test set from the test set.
    # Start
    X, y, X_y_df = extract_elements_from_2Ddict(test_set)
    X_y_df.to_csv(os.path.join(TARGET_PATH, 'test_documents.csv'))
    # end
    del test_set
    dficf.fit_transform(X, is_random=True)
    # print('test_set_voc_', dficf.new_doc_voc_)
    # print('test_set_terms', dficf.term_dic_for_each_document)
    # print('transformed_input_data_for_each_category_voc_', dficf.transformed_input_data_for_each_category_voc_)
    X_data, y_data, y_data_, sentence_index, all_together = dficf.vectors_
    # print('vectors of test set', dficf.vectors_)
    # print('categories as dict', dficf.category_as_dict_)

    test_data_all_in_one= pd.DataFrame(data=all_together, columns=['vector', 'generated_category_indx', 'generated_category_name', 'actual_index'])

    test_data_all_in_one['actual_category_name'] = test_data_all_in_one.apply(lambda  row: y[row['actual_index']], axis=1)
    test_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'dficf_test_all_in_one.csv'))

    test_data_df = pd.DataFrame(data=y_data_, columns=["Cat"])
    test_data_df.to_csv(os.path.join(TARGET_PATH, 'dficf_test_labels.csv'))
    util.dump_adff('DFICF', 'DFICF BBC DATA', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)),
                   dficf.category_as_dict_, X_data, y_data_, TARGET_PATH, 'dficf_test_bbc_arff', False)

    #
    # print('results')
    # print('(max is important) score based:', dficf.similarity_scores_)
    # print('(min is important) euclidean distance based:', dficf.similarity_euclidean_distance_)
    # print('(max is important) cosine distance based:', dficf.similarity_cosine_distance_)

    # if save is True:
    #     with open(os.path.join(TARGET_PATH, 'dficf_weighs_for_documents_in_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.dficf_weighs_for_documents_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(os.path.join(TARGET_PATH,'vocabulary_for_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.vocabulary_for_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(os.path.join(TARGET_PATH, 'dcicf_weights_for_terms_in_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.dficf_weighs_for_terms_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def classical_tfidf(data_size_rate, is_test_run=False, save=False):
    # ORIGINAL LIB, MISSING LEMMATIZATON
    # from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    # COPY OF ORIGINAL BUT HAS A LEMMATIZATION MODULE
    from innovations.dficf.copy_of_orj_tfidf_vectorizer import CountVectorizer, TfidfTransformer

    # NEW WAY TO DO THAT
    if is_test_run is False:
        start_time = dt.datetime.now()
        print('Files are getting ready')
        root_folder_name, files_dataframe = get_files_content(TARGET_PATH, 'txt')
        print('{} of files are parsed in {} mins.'.format(str(len(files_dataframe)),
                                                          ((dt.datetime.now() - start_time).seconds) / 60))
        np.random.seed(0)
        data = files_dataframe.groupby(['Procedure'])
        len_of_data = len(data)
        train_set = collections.OrderedDict()
        test_set = collections.OrderedDict()
        for _proc, _grouped in data:
            doc_size = len(_grouped.Features)
            print("Data size of {} proc is {}".format(_proc, doc_size))
            print("With adding an data rate {}".format(int(doc_size / data_size_rate)))
            docs = _grouped.Features.transform(np.random.permutation)
            features_train, features_test = train_test_split(docs[:int(doc_size / data_size_rate)], test_size=0.33)
            train_set[_proc] = list()
            # train_set[_proc].extend([x for x in _grouped.Features])
            train_set[_proc].extend([x for x in features_train])
            test_set[_proc] = list()
            test_set[_proc].extend([x for x in features_test])

    else:
        train_set = [
            "Katerina Thanou is confident she and fellow sprinter Kostas Kenteris will not be punished for missing drugs tests before the Athens Olympics."
            "The Greek pair appeared at a hearing on Saturday which will determine whether their provisional bans from athletics ruling body the IAAF should stand. "
            "After five months we finally had the chance to give explanations. I am confident and optimistic, said Thanou. The athletes lawyer Grigoris Ioanidis "
            "said he believed the independent disciplinary committee set up by the Greek Athletics Federation would find them innocent. We are almost certain that "
            "the charges will be dropped, said Ioanidis.",

            "London Marathon organisers are hoping that banned athlete Susan Chepkemei will still take part in this year's race on 17 April.Chepkemei was suspended from all competition "
            "until the end of the year by Athletics Kenya after failing to report to a national training camp. We are watching it closely said London race director David Bedford. The camp "
            "in Embu was to prepare for the IAAF World Cross Country Championships later this month. Chepkemei however took part and finished third in last Sunday's world best 10K race in "
            "Puerto Rico.",

            "Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross-Country Championships. The 31-year-old is concerned the event, which "
            "starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. Radcliffe was world cross-country champion in 2001 and 2002 but missed "
            "last year's event because of injury. In her absence, the GB team won bronze in Brussels.",

            "Wayne Rooney made a winning return to Everton as Manchester United cruised into the FA Cup quarter-finals."
            "Rooney received a hostile reception, but goals in each half from Quinton Fortune and Cristiano Ronaldo silenced the jeers at Goodison Park. Fortune headed home after "
            "23 minutes before Ronaldo scored when Nigel Martyn parried Paul Scholes' free-kick. Marcus Bent missed Everton's best chance when Roy Carroll, who was later struck by "
            "a missile, saved at his feet. It was a fantastic performance by us. In fairness I think Everton have missed a couple of players and got some young players out.",

            "A brave defensive display, led by keeper David James, helped Manchester City hold the leaders Chelsea. After a quiet opening, James denied Damien Duff, Jiri Jarosik and "
            "Mateja Kezman, while Paul Bosvelt cleared William Gallas' header off the line. Robbie Fowler should have scored for the visitors but sent his header wide. Chelsea had "
            "most of the possession in the second half but James kept out Frank Lampard's free-kick and superbly tipped the same player's volley wide. City went into the game with the "
            "proud record of being the only domestic team to beat Chelsea this season.",

            "Before now many spammers have recruited home PCs to act as anonymous e-mail relays in an attempt to hide the origins of their junk mail. The PCs are recruited using viruses "
            "and worms that compromise machines via known vulnerabilities or by tricking people into opening an attachment infected with the malicious program. Once compromised the machines "
            "start to pump out junk mail on behalf of spammers. Spamhaus helps to block junk messages from these machines by collecting and circulating blacklists of net addresses known to "
            "harbour infected machines. But the novel worm spotted recently by Spamhaus routes junk via the mail servers of the net service firm that infected machines used to get online in "
            "the first place. In this way the junk mail gets a net address that looks legitimate. As blocking all mail from net firms just to catch the spam is impractical, Spamhaus is worried "
            "that the technique will give junk mailers the ability to spam with little fear of being spotted and stopped.",

            "Use of games in schools has been patchy she found, with Sim City proving the most popular. Traditionally schools have eschewed mainstream games in favour of used so-called edu-tainment "
            "software in a belief that such packages help to make learning fun, she found in her research. It is perhaps in a compromise between edutainment and mainstream games that the greatest "
            "potential for classroom useable games lies, she wrote in a paper entitled Games and Learning. 'Lite' versions of existing games could be the way forward and would overcome one of the "
            "biggest hurdles - persuading developers to write for the educational market. This would appeal to developers because of the low costs involved in adapting them as well as offering a "
            "new opportunity for marketing. Already there are games on the market, such as Civilisation and Age of Empire, that have educational elements said Mr Owen. Even in Grand Theft Auto it "
            "is not just the violence that engages people, he said. It could be some time until that particular game makes it into the classroom though.",

            "Microsoft's Media Player 10 took the award for Most Wanted Software. This year was the 10th anniversary of the PC Pro awards, which splits its prizes into two sections. The first are "
            "chosen by the magazine's writers and consultants, the second are voted for by readers. Mr Trotter said more than 13,000 people voted for the Reliability and Service Awards, "
            "twice as many as in 2003. Net-based memory and video card shop Crucial shared the award for Online Vendor of the year with Novatech.",

            "If the ratings are confirmed, the episode will have given the soap its highest audience for a year. The overnight figures showed almost 60% of the viewing public tuned into EastEnders "
            "between 2000 and 2100 GMT, leaving ITV1 with about 13%. We are very pleased with the figures,a BBC spokesman said. It shows viewers have really enjoyed the story of Den's demise. "
            "The show's highest audience came at Christmas 1986, when more than 30 million tuned in to see Den, played by Leslie Grantham, hand divorce papers to wife Angie.",

            "Sky first scooped Oscar rights from the BBC in 1999, but the BBC won them back in 2001 when Sky was forced to pull out of a bidding war due to financial constraints. BBC viewers will "
            "of course be able to watch quality coverage of the 2005 Academy Awards on the BBC's bulletins and news programmes, a spokesman said. Among the films tipped to do well at this year's "
            "Academy Awards are Martin Scorsese's The Aviator, Jean-Pierre Jeunet's A Very Long Engagement and the Ray Charles biopic, Ray"]
        test_set = ['football season is getting started']

    count_vectorizer = CountVectorizer(stop_words='english', stemming_and_lemmatization=True)

    X_train, y_train, x_y_df = extract_elements_from_2Ddict(train_set)
    del train_set

    count_vectorizer.fit_transform(X_train)
    print("Vocabulary:", count_vectorizer.vocabulary_)
    freq_term_matrix = count_vectorizer.transform(X_train)
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.transform(freq_term_matrix)
    train_docs_with_tfidf = tf_idf_matrix.toarray()
    print("TFIDF Train:", train_docs_with_tfidf)

    # dictionary?
    my_dict = {i: k for i, k in enumerate(set(y_train))}

    all_together = list(zip(train_docs_with_tfidf.tolist(), y_train.tolist()))
    train_data_all_in_one = pd.DataFrame(data=all_together,
                                        columns=['vector', 'generated_category_name'])
    train_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'tfidf_train_all_in_one.csv'))

    util.dump_adff('TFIDF', 'TFIDF BBC DATA',
                   count_vectorizer.sort_voc(count_vectorizer.reverse_voc(count_vectorizer.vocabulary_)),
                   my_dict, train_docs_with_tfidf, y_train, TARGET_PATH, 'tfidf_train_bbc_arff')

    X_test, y_test, x_y_df = extract_elements_from_2Ddict(test_set)
    del test_set
    freq_term_matrix = count_vectorizer.transform(X_test)
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    tf_idf_matrix = tfidf.transform(freq_term_matrix)
    test_docs_with_tfidf = tf_idf_matrix.toarray()
    print("TFIDF Test:", test_docs_with_tfidf)

    all_together = list(zip(test_docs_with_tfidf.tolist(),y_test.tolist()))

    test_data_all_in_one = pd.DataFrame(data=all_together,
                                        columns=['vector', 'generated_category_name'])
    test_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'tfidf_test_all_in_one.csv'))

    test_data_df = pd.DataFrame(data=y_test, columns=["Cat"])
    test_data_df.to_csv(os.path.join(TARGET_PATH, 'tfidf_test_labels.csv'))

    util.dump_adff('TFIDF', 'TFIDF BBC DATA', count_vectorizer.sort_voc(
        count_vectorizer.reverse_voc(count_vectorizer.vocabulary_)),
                   my_dict, test_docs_with_tfidf, y_test, TARGET_PATH, 'tfidf_test_bbc_arff', False)
    # if is_test_run is False:

    scores = []
    # for _test_doc in test_docs_with_tfidf:
    #     final_new_document_distance_score = 0
    #     score_details = collections.OrderedDict()
    #     i = 0
    #     for _doc in train_docs_with_tfidf:
    #         _new_document_distance_score = 1 - spatial.distance.cosine(_doc, _test_doc)
    #         _new_document_distance_score = (
    #             0 if math.isnan(_new_document_distance_score) is True else _new_document_distance_score)
    #         if _new_document_distance_score > final_new_document_distance_score:
    #             final_new_document_distance_score = _new_document_distance_score
    #             final_i = i
    #         i += 1
    #     score_details['score'] = final_new_document_distance_score
    #     if is_test_run is False:
    #         score_details['doc_indx'] = final_i
    #         score_details['category'] = files_dataframe.iloc[final_i][0]
    #     else:
    #         score_details['category'] = ('sport' if final_i <= 4 else 'tech' if final_i <= 7 else 'entertainment')
    #     scores.append(score_details)
    #
    # print("Scores cosine distance:", scores)
    #
    # if is_test_run is False:
    #     print('sample index data from train set', files_dataframe.values[final_i])
    #
    # if save is True:
    #     with open(os.path.join(TARGET_PATH, 'tfidf_weighs_for_each_document.pickle'), 'wb') as handle:
    #         pickle.dump(tf_idf_matrix.toarray(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(os.path.join(TARGET_PATH, 'vocabulary_for_corpus.pickle'), 'wb') as handle:
    #         pickle.dump(count_vectorizer.vocabulary_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_elements_from_2Ddict(_2d_dict):
    X_y = []
    for _key, _values in _2d_dict.items():
        for _element in _values:
            X_y.append((_element, _key))

    # np.random.seed(0)
    random.shuffle(X_y)
    df = pd.DataFrame(data=X_y, columns=['Doc', 'Cat'])
    X = df.Doc.values
    y = df.Cat.values
    del X_y
    del _2d_dict
    return X, y, df



def hiarchical_tficf(data_size_rate, is_test_run=False, save=False):
    from innovations.dficf.modified_dficf_vectorizer_added_tf import CountVectorizer, TfidfTransformer, DfIcfTransformer

    # NEW WAY TO DO THAT
    if is_test_run is False:
        start_time = dt.datetime.now()
        print('Files are getting ready')
        root_folder_name, files_dataframe = get_files_content(TARGET_PATH, 'txt')
        print('{} of files are parsed in {} mins.'.format(str(len(files_dataframe)),
                                                          ((dt.datetime.now() - start_time).seconds) / 60))
        np.random.seed(0)
        data = files_dataframe.groupby(['Procedure'])
        len_of_data = len(data)
        train_set = collections.OrderedDict()
        test_set = collections.OrderedDict()
        for _proc, _grouped in data:
            doc_size = len(_grouped.Features)
            print("Data size of {} proc is {}".format(_proc, doc_size))
            print("With adding an data rate {}".format(int(doc_size / data_size_rate)))
            docs = _grouped.Features.transform(np.random.permutation)
            features_train, features_test = train_test_split(docs[:int(doc_size / data_size_rate)], test_size=0.33)
            train_set[_proc] = list()
            # train_set[_proc].extend([x for x in _grouped.Features])
            train_set[_proc].extend([x for x in features_train])
            test_set[_proc] = list()
            test_set[_proc].extend([x for x in features_test])

    else:
        train_set = collections.OrderedDict()
        # test_set = dict()

        train_set['sport'] = [
            # "Katerina Thanou is confident she and fellow sprinter Kostas Kenteris will not be punished for missing drugs tests before the Athens Olympics."
            # "The Greek pair appeared at a hearing on Saturday which will determine whether their provisional bans from athletics ruling body the IAAF should stand. "
            # "After five months we finally had the chance to give explanations. I am confident and optimistic, said Thanou. The athletes lawyer Grigoris Ioanidis "
            # "said he believed the independent disciplinary committee set up by the Greek Athletics Federation would find them innocent. We are almost certain that "
            # "the charges will be dropped, said Ioanidis.",
            #
            # "London Marathon organisers are hoping that banned athlete Susan Chepkemei will still take part in this year's race on 17 April.Chepkemei was suspended from all competition "
            # "until the end of the year by Athletics Kenya after failing to report to a national training camp. We are watching it closely said London race director David Bedford. The camp "
            # "in Embu was to prepare for the IAAF World Cross Country Championships later this month. Chepkemei however took part and finished third in last Sunday's world best 10K race in "
            # "Puerto Rico.",
            #
            # "Paula Radcliffe has been granted extra time to decide whether to compete in the World Cross-Country Championships. The 31-year-old is concerned the event, which "
            # "starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. Radcliffe was world cross-country champion in 2001 and 2002 but missed "
            # "last year's event because of injury. In her absence, the GB team won bronze in Brussels.",
            #
            # "Wayne Rooney made a winning return to Everton as Manchester United cruised into the FA Cup quarter-finals."
            # "Rooney received a hostile reception, but goals in each half from Quinton Fortune and Cristiano Ronaldo silenced the jeers at Goodison Park. Fortune headed home after "
            # "23 minutes before Ronaldo scored when Nigel Martyn parried Paul Scholes' free-kick. Marcus Bent missed Everton's best chance when Roy Carroll, who was later struck by "
            # "a missile, saved at his feet. It was a fantastic performance by us. In fairness I think Everton have missed a couple of players and got some young players out.",
            #
            # "A brave defensive display, led by keeper David James, helped Manchester City hold the leaders Chelsea. After a quiet opening, James denied Damien Duff, Jiri Jarosik and "
            # "Mateja Kezman, while Paul Bosvelt cleared William Gallas' header off the line. Robbie Fowler should have scored for the visitors but sent his header wide. Chelsea had "
            # "most of the possession in the second half but James kept out Frank Lampard's free-kick and superbly tipped the same player's volley wide. City went into the game with the "
            # "proud record of being the only domestic team to beat Chelsea this season."
            "football ball player", "ball player"
        ]

        train_set['tech'] = [
            # "Before now many spammers have recruited home PCs to act as anonymous e-mail relays in an attempt to hide the origins of their junk mail. The PCs are recruited using viruses "
            # "and worms that compromise machines via known vulnerabilities or by tricking people into opening an attachment infected with the malicious program. Once compromised the machines "
            # "start to pump out junk mail on behalf of spammers. Spamhaus helps to block junk messages from these machines by collecting and circulating blacklists of net addresses known to "
            # "harbour infected machines. But the novel worm spotted recently by Spamhaus routes junk via the mail servers of the net service firm that infected machines used to get online in "
            # "the first place. In this way the junk mail gets a net address that looks legitimate. As blocking all mail from net firms just to catch the spam is impractical, Spamhaus is worried "
            # "that the technique will give junk mailers the ability to spam with little fear of being spotted and stopped.",
            #
            # "Use of games in schools has been patchy she found, with Sim City proving the most popular. Traditionally schools have eschewed mainstream games in favour of used so-called edu-tainment "
            # "software in a belief that such packages help to make learning fun, she found in her research. It is perhaps in a compromise between edutainment and mainstream games that the greatest "
            # "potential for classroom useable games lies, she wrote in a paper entitled Games and Learning. 'Lite' versions of existing games could be the way forward and would overcome one of the "
            # "biggest hurdles - persuading developers to write for the educational market. This would appeal to developers because of the low costs involved in adapting them as well as offering a "
            # "new opportunity for marketing. Already there are games on the market, such as Civilisation and Age of Empire, that have educational elements said Mr Owen. Even in Grand Theft Auto it "
            # "is not just the violence that engages people, he said. It could be some time until that particular game makes it into the classroom though.",
            #
            # "Microsoft's Media Player 10 took the award for Most Wanted Software. This year was the 10th anniversary of the PC Pro awards, which splits its prizes into two sections. The first are "
            # "chosen by the magazine's writers and consultants, the second are voted for by readers. Mr Trotter said more than 13,000 people voted for the Reliability and Service Awards, "
            # "twice as many as in 2003. Net-based memory and video card shop Crucial shared the award for Online Vendor of the year with Novatech."
            "football game console", "game console"
        ]

        train_set['entertainment'] = [
            # "If the ratings are confirmed, the episode will have given the soap its highest audience for a year. The overnight figures showed almost 60% of the viewing public tuned into EastEnders "
            # "between 2000 and 2100 GMT, leaving ITV1 with about 13%. We are very pleased with the figures,a BBC spokesman said. It shows viewers have really enjoyed the story of Den's demise. "
            # "The show's highest audience came at Christmas 1986, when more than 30 million tuned in to see Den, played by Leslie Grantham, hand divorce papers to wife Angie.",
            #
            # "Sky first scooped Oscar rights from the BBC in 1999, but the BBC won them back in 2001 when Sky was forced to pull out of a bidding war due to financial constraints. BBC viewers will "
            # "of course be able to watch quality coverage of the 2005 Academy Awards on the BBC's bulletins and news programmes, a spokesman said. Among the films tipped to do well at this year's "
            # "Academy Awards are Martin Scorsese's The Aviator, Jean-Pierre Jeunet's A Very Long Engagement and the Ray Charles biopic, Ray."
            "cinema console actor", "cinema actor"
        ]

        test_set = ['football season is getting started']

        print('train set:', train_set)
        print('test set:', test_set)

    count_vectorizer = CountVectorizer(stop_words='english', stemming_and_lemmatization=True)

    # categorical values from the trained count vectorizer.
    vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies, dict_for_number_of_documents, \
    dict_for_local_category_document_term_count, dict_for_global_category_document_term_count, \
    dict_for_global_category_count = count_vectorizer.fit_transform(train_set)

    del count_vectorizer

    dficf = DfIcfTransformer(vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies,
                             dict_for_number_of_documents,
                             dict_for_local_category_document_term_count, dict_for_global_category_document_term_count,
                             dict_for_global_category_count)

    dficf.transform()
    # print('Merged vocabularies:', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)))
    # for _cat, _val in dficf.dficf_weighs_for_documents_in_each_category_.items():
    #     print("-" * 50)
    #     print(_cat)
    #     print("-" * 50)
    #
    #     # print(10 * '*')
    #     # print("dficf_weighs_for_documents_in_each_category:", _val.toarray())
    #     #
    #     # print(10 * '*')
    #     # print("vocabulary_for_each_category")
    #     # for _subkey, _subval in dficf.vocabulary_for_each_category_[_cat].items():
    #     #     print('Term ({}) : Position/Index ({})'.format(_subkey, _subval))
    #     # print(10 * '*')
    #     # print("dcicf_weights_for_terms_in_each_category")
    #     # for _subkey, _subval in dficf.dficf_weighs_for_terms_in_each_category_[_cat].items():
    #     #     print('Term ({}) : dficf weight ({})'.format(_subkey, _subval))
    #     _voc = dficf.vocabulary_for_each_category_[_cat]
    #     _reversed = dficf.reversed_vocabulary_for_each_category_[_cat]
    #     _sorted_reversed = dficf.reversed_ordered_vocabulary_for_each_category_[_cat]
    #
    #     print('Normal Voc:', _voc)
    #     print('Reversed Voc:', _reversed)
    #     print('Sorted Reversed Voc:', _sorted_reversed)

    # Just add the known items to the test set from the test set.
    # Start
    X, y, X_y_df = extract_elements_from_2Ddict(train_set)
    X_y_df.to_csv(os.path.join(TARGET_PATH, 'train_documents.csv'))
    # end
    del train_set
    dficf.fit_transform(X)
    # print('test_set_voc_', dficf.new_doc_voc_)
    # print('test_set_terms', dficf.term_dic_for_each_document)
    # print('transformed_input_data_for_each_category_voc_', dficf.transformed_input_data_for_each_category_voc_)
    X_data, y_data, y_data_, sentence_index, all_together = dficf.vectors_


    # print('vectors of test set', dficf.vectors_)
    # print('categories as dict', dficf.category_as_dict_)

    # Just add the known items to the test set from the test set.
    # Start
    len_of_test = len(X)
    len_of_test_ = len(X_data)
    X_data_ = []
    y_data__ = []
    for _i, _y in enumerate(y):
        print('*' * 10)
        print('Document index:', str(_i))
        for _i_vectors in range(_i, len_of_test_, len_of_test):
            print('Checking index in Data:', str(_i_vectors))
            if (_y == y_data_[_i_vectors]):
                print('Found index in Data:', str(_i_vectors))
                print('Found doc in Data:', _y)
                X_data_.append(X_data[_i_vectors])
                y_data__.append(y_data_[_i_vectors])
                break

    # End
    all_together = list(zip(X_data_, y_data__))
    train_data_all_in_one = pd.DataFrame(data=all_together,
                                         columns=['vector', 'generated_category_name'])

    train_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'dficf_train_all_in_one.csv'))

    util.dump_adff('DFICF', 'DFICF BBC DATA', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)),
                   dficf.category_as_dict_, X_data_, y_data__, TARGET_PATH, 'dficf_train_bbc_arff')

    # Just add the known items to the test set from the test set.
    # Start
    X, y, X_y_df = extract_elements_from_2Ddict(test_set)
    X_y_df.to_csv(os.path.join(TARGET_PATH, 'test_documents.csv'))
    # end
    del test_set
    dficf.fit_transform(X, is_random=True)
    # print('test_set_voc_', dficf.new_doc_voc_)
    # print('test_set_terms', dficf.term_dic_for_each_document)
    # print('transformed_input_data_for_each_category_voc_', dficf.transformed_input_data_for_each_category_voc_)
    X_data, y_data, y_data_, sentence_index, all_together = dficf.vectors_
    # print('vectors of test set', dficf.vectors_)
    # print('categories as dict', dficf.category_as_dict_)

    test_data_all_in_one= pd.DataFrame(data=all_together, columns=['vector', 'generated_category_indx', 'generated_category_name', 'actual_index'])

    test_data_all_in_one['actual_category_name'] = test_data_all_in_one.apply(lambda  row: y[row['actual_index']], axis=1)
    test_data_all_in_one.to_csv(os.path.join(TARGET_PATH, 'dficf_test_all_in_one.csv'))

    test_data_df = pd.DataFrame(data=y_data_, columns=["Cat"])
    test_data_df.to_csv(os.path.join(TARGET_PATH, 'dficf_test_labels.csv'))
    util.dump_adff('DFICF', 'DFICF BBC DATA', dficf.sort_voc(dficf.reverse_voc(dficf.merged_vocabularies_)),
                   dficf.category_as_dict_, X_data, y_data_, TARGET_PATH, 'dficf_test_bbc_arff', False)

    #
    # print('results')
    # print('(max is important) score based:', dficf.similarity_scores_)
    # print('(min is important) euclidean distance based:', dficf.similarity_euclidean_distance_)
    # print('(max is important) cosine distance based:', dficf.similarity_cosine_distance_)

    # if save is True:
    #     with open(os.path.join(TARGET_PATH, 'dficf_weighs_for_documents_in_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.dficf_weighs_for_documents_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(os.path.join(TARGET_PATH,'vocabulary_for_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.vocabulary_for_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(os.path.join(TARGET_PATH, 'dcicf_weights_for_terms_in_each_category.pickle'), 'wb') as handle:
    #         pickle.dump(dficf.dficf_weighs_for_terms_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)

# SCIKIT LEARN STYLE OF MY CUSTOM TFIDF
if __name__ == "__main__":
    # prod
    data_size_rate = 2.5
    #classical_tfidf(data_size_rate=data_size_rate)
    #hiarchical_dficf(data_size_rate=data_size_rate)

    #hiarchical_tficf(data_size_rate=data_size_rate)
    classical_tfidf(data_size_rate=data_size_rate)

    # test
    # hiarchical_dficf(True)
    # classical_tfidf(True)

# start_time = dt.datetime.now()
# print('Files are getting ready')
# root_folder_name, files_dataframe = get_files_content(TARGET_PATH, 'txt')
# print('{} of files are parsed in {} mins.'.format(str(len(files_dataframe)),
#                                                   ((dt.datetime.now() - start_time).seconds) / 60))
#
#
# #NEW WAY TO DO THAT
# from ahhead_ai.experiments.text import CountVectorizer, TfidfTransformer, DfIcfTransformer
# # data = files_dataframe.groupby(['Procedure'])
# # number_of_accounts = len(data)
# # train_set = dict()
# # test_set = dict()
# # for _proc, _grouped in data:
# #     train_set[_proc] = list()
# #     train_set[_proc].extend([x for x in _grouped.Features])
# #     # train_set[_proc].extend([x for x in _grouped.Features[:25]])
# #     # test_set[_proc] = list()
# #     # test_set[_proc].extend([x for x in _grouped.Features[30:]])
# # i = 0
#
# train_set = dict()
# # test_set = dict()
# train_set['sport'] = ["galatasaray is champion team", "fenerbahce has some problems as a team"]
# train_set['finance'] = ["oil is expensive, money is not enough for oil team", "galatasaray stock is expensive"]
# i = 0
#
# #EXISTING WAY TO DO THAT
# #from ahhead_ai.experiments.orj_text import CountVectorizer, TfidfTransformer
# # train_set = [x for x in files_dataframe.Features[:25]]
# # test_set = [x for x in files_dataframe.Features[30:]]
# #i = 1
#
# #
# # train_set = ["The sky is blue.", "The sun is bright seen rock n roll.","sky ,rock"]
# # test_set = ["The sun in the sky is bright.",
# #             "We can see the shining sun, the bright sun."]
#
#
# count_vectorizer = CountVectorizer(stop_words='english',  stemming_and_lemmatization=True )
#
# #categorical values from the trained count vectorizer.
# vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies, dict_for_number_of_documents, \
# dict_for_local_category_document_term_count, dict_for_global_category_document_term_count, \
# dict_for_global_category_count = count_vectorizer.fit_transform(train_set)
#
# #count_vectorizer.fit_transform(train_set)
# #print('Train set:', train_set)
# #print('Test set:', test_set)
#
# if i < 1:
#     dficf = DfIcfTransformer(vocabularies_with_categories_, number_of_categories, dict_for_term_frequencies, dict_for_number_of_documents,
#                              dict_for_local_category_document_term_count, dict_for_global_category_document_term_count,
#                              dict_for_global_category_count).fit_transform()
#
#     dficf.fit_transform()
#     for _cat, _val in dficf.dficf_weighs_for_documents_in_each_category_.items():
#         print("-" * 50)
#         print(_cat)
#         print("-" * 50)
#
#         print(10 * '*')
#         print("dficf_weighs_for_documents_in_each_category:", _val.toarray())
#
#         print(10 * '*')
#         print("vocabulary_for_each_category")
#         for _subkey,_subval in dficf.vocabulary_for_each_category_[_cat].items():
#             print('Term ({}) : Position/Index ({})'.format(_subkey, _subval))
#         print(10 * '*')
#         print("dcicf_weights_for_terms_in_each_category")
#         for _subkey, _subval in dficf.dficf_weighs_for_terms_in_each_category_[_cat].items():
#             print('Term ({}) : dficf weight ({})'.format(_subkey, _subval))
#
#     # with open(os.path.join(TARGET_PATH, 'dficf_weighs_for_documents_in_each_category.pickle'), 'wb') as handle:
#     #     pickle.dump(dficf.dficf_weighs_for_documents_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     #
#     # with open(os.path.join(TARGET_PATH,'vocabulary_for_each_category.pickle'), 'wb') as handle:
#     #     pickle.dump(dficf.vocabulary_for_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     #
#     # with open(os.path.join(TARGET_PATH, 'dcicf_weights_for_terms_in_each_category.pickle'), 'wb') as handle:
#     #     pickle.dump(dficf.dficf_weighs_for_terms_in_each_category_, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#     # freq_term_matrix = count_vectorizer.transform(test_set)
#     # for _category, _matrix in freq_term_matrix.items():
#     #     print(_category, '- Dense:', _matrix.todense())
#
# else:
#     print("Vocabulary:", count_vectorizer.vocabulary_)
#     freq_term_matrix = count_vectorizer.transform(test_set)
#     print('Dense:', freq_term_matrix.todense())
#
#     tfidf = TfidfTransformer(norm="l2")
#     tfidf.fit(freq_term_matrix)
#     print("IDF:", tfidf.idf_)
#
#     tf_idf_matrix = tfidf.transform(freq_term_matrix)
#     print("TF_IDF:", tf_idf_matrix.todense())

# MY STYLE OF MY CUSTOM TFIDF
'''
if __name__ == "__main__":
    start_time = dt.datetime.now()
    print('Files are getting ready')
    root_folder_name, files_dataframe = get_files_content(TARGET_PATH, 'txt')
    print('{} of files are parsed in {} mins.'.format(str(len(files_dataframe)),
                                                      ((dt.datetime.now() - start_time).seconds) / 60))

    # columns=['root_folder', 'file_category', 'file_name', 'file_path','file_content'])

    start_time = dt.datetime.now()
    print('T-TFIDF has started')
    accounts_per_proc_data_columns = ['Procedure', 'AccountID', 'Features']
    temp_df = files_dataframe.copy()
    accounts_per_proc_data = temp_df[accounts_per_proc_data_columns]
    accounts_per_proc_data['Procedure'] = accounts_per_proc_data.Procedure.astype(str)
    accounts_per_proc_data['AccountID'] = accounts_per_proc_data.AccountID.astype(str)
    accounts_per_proc_data['Features'] = accounts_per_proc_data.Features.astype(str)

    # -------------------------------------------------------------------------
    # accounts are grouped per proc
    # -------------------------------------------------------------------------
    grouped_account_per_proc_data = accounts_per_proc_data.groupby(['Procedure'])

    # --------------------------------------------------------
    # TF IDF : OTHER PROC ACCOUNTS DATA EXCLUDING CURRENT LOCAL PROC AND ITS
    # --------------------------------------------------------
    # accounts are grouped per proc-acct
    # -------------------------------------------------------------------------
    account_per_proc_for_group_data = accounts_per_proc_data.copy()
    account_per_proc_for_group_data = account_per_proc_for_group_data.drop(['AccountID'], axis=1)

    # grouped_account_per_proc_and_account_data_excluding_cur_proc = account_per_proc_for_group_data.ix[
    #     account_per_proc_for_group_data['Procedure'] != _proc_code]

    account_per_proc_for_group_data = account_per_proc_for_group_data.groupby('Procedure')['Features'].apply(set)
    other_procs_group_total_accounts = len(account_per_proc_for_group_data) - 1
    other_procs_group_features = [' '.join(document).split() for document in
                                  account_per_proc_for_group_data]
    # other_procs_group_features = [document.split() for document in
    #                               account_per_proc_for_group_data]
    other_procs_group_count_dict = collections.Counter(
        chain.from_iterable(set(x) for x in other_procs_group_features))

    # ------------------------------------------------------------------------
    # ACCOUNT IS DOCUMENT -> LOCAL FREQUENCY
    # ------------------------------------------------------------------------

    number_of_records_in_grouped_account_per_proc_data = len(grouped_account_per_proc_data)
    global_index = 0
    old_percent = 0
    for _header, _grouped in grouped_account_per_proc_data:
        _proc_code = _header

        proc_timer_start = dt.datetime.now()

        print(50 * ":")
        print("Proc: {}".format(_proc_code))
        data_account_features = []
        for _size_i in range(0, len(_grouped)):
            _accountid = _grouped.AccountID.values[_size_i]
            _features = _grouped.Features.values[_size_i]
            data_account_features.append(
                (_proc_code, _accountid, _features))

        # proc_data raw file exporting
        proc_data_file_name = _proc_code
        proc_data = pd.DataFrame(data=data_account_features, columns=['Procedure', 'AccountID', 'Features'])
        proc_data.to_csv(os.path.join(TARGET_PATH, proc_data_file_name + '.csv'), header=True, index=True,
                         sep='|')

        # --------------------------------------------------------
        # TF IDF : CURRENT LOCAL PROC ACCOUNTS DATA
        # --------------------------------------------------------
        current_proc_total_accounts = len(proc_data)
        current_proc_features = [document.split() for document in proc_data['Features']]
        current_proc_doc_count_dict = collections.Counter(chain.from_iterable(set(x) for x in current_proc_features))
        current_proc_term_count_dict = collections.Counter(chain.from_iterable(x for x in current_proc_features))

        # --------------------------------------------------------
        # TF IDF : OTHER PROC ACCOUNTS DATA EXCLUDING CURRENT LOCAL PROC AND ITS
        # --------------------------------------------------------
        other_procs_data = accounts_per_proc_data.ix[accounts_per_proc_data['Procedure'] != _proc_code]
        other_procs_total_accounts = len(other_procs_data)
        other_procs_features = [document.split() for document in other_procs_data['Features']]
        other_procs_doc_count_dict = collections.Counter(chain.from_iterable(set(x) for x in other_procs_features))

        # --------------------------------------------------------
        # TF IDF : CLASS
        # --------------------------------------------------------
        tfidf_obj = hierarchical_tfidf.TFIDF(proc_data,
                                             current_proc_total_accounts,
                                             current_proc_doc_count_dict,
                                             current_proc_term_count_dict,
                                             other_procs_total_accounts,
                                             other_procs_doc_count_dict,
                                             other_procs_group_total_accounts,
                                             other_procs_group_count_dict)
        tfidf_obj.execute()
        tf_idf_result_dataframe = tfidf_obj.result_as_dataframe
        # Calculated TFIDF File
        tf_idf_result_dataframe = tf_idf_result_dataframe.sort_values(by=['Term', 'GOD_TFxDFxIDFxIGF'],
                                                                      ascending=[True, False])
        tf_idf_result_dataframe.to_csv(os.path.join(TARGET_PATH, proc_data_file_name + '_calculated_tfidf.csv'),
                                       header=True)

        clean_data_group = tf_idf_result_dataframe.drop_duplicates(['Term'], keep='first')
        clean_data_group = clean_data_group[['Term', 'GOD_DFxIDFxIGF']]
        clean_data_group = clean_data_group.sort_values(by=['GOD_DFxIDFxIGF'],
                                                        ascending=[False])
        clean_data = []
        for index, _row in clean_data_group.iterrows():
            clean_data.append((_row['Term'], _row['GOD_DFxIDFxIGF']))

        clean_data_df = pd.DataFrame(data=clean_data, columns=['Term', 'GOD_DFxIDFxIGF'])
        clean_data_df.to_csv(os.path.join(TARGET_PATH, proc_data_file_name + '_clean_order.csv'),
                             header=True)

        percent = round(((global_index + 1) / number_of_records_in_grouped_account_per_proc_data) * 100)
        if old_percent != percent:
            print("{} % of data processed".format(str(percent)))
            old_percent = percent

        global_index += 1


        # tfidf_with_tolgahan_method_category_based_idf(root_folder_name, files_dataframe)

    print('{} of files are completed in {} mins.'.format(str(len(files_dataframe)),
                                                         ((dt.datetime.now() - start_time).seconds) / 60))


    # print(os.getcwd())
    # print(tfidf(files_dataframe))
'''
