from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import re
import numpy as np
from Loader import Loader


class DataAnalyzer:

    def __init__(self, loader: Loader):
        self.loader = loader

    def get_3class_frequency_occurrence(self):
        ratings = self.loader.get_ratings_values()
        r0_0_4 = sum(map(lambda el: el <= 0.4, ratings))
        r0_4_0_7 = sum(map(lambda el: 0.4 < el < 0.7, ratings))
        r0_7_1 = sum(map(lambda el: el >= 0.7, ratings))

        x_label = ['r <= 0.4', '0.4 < r < 0.7', 'r >= 0.7']
        y_label = [r0_0_4, r0_4_0_7, r0_7_1]

        fig, ax = plt.subplots()
        ax.bar(x_label, y_label)

        plt.show()

    def get_4class_frequency_occurrence(self):
        ratings = self.loader.get_ratings_values()
        r0_0_3 = sum(map(lambda el: el <= 0.43, ratings))
        r0_4_0_5 = sum(map(lambda el: 0.4 <= el <= 0.5, ratings))
        r0_6_0_7 = sum(map(lambda el: 0.6 <= el <= 0.7, ratings))
        r0_8_1 = sum(map(lambda el: el >= 0.8, ratings))

        x_label = ['r <= 0.3', '0.4 < r < 0.5', '0.6 < r < 0.7', 'r >= 0.8']
        y_label = [r0_0_3, r0_4_0_5, r0_6_0_7, r0_8_1]

        fig, ax = plt.subplots()
        ax.bar(x_label, y_label)

        plt.show()

    def get_review_length(self):
        dictionary = {}
        for text in self.loader.get_reviews_values():
            count = len(re.findall(r'\w+', text))
            if count not in dictionary:
                dictionary[count] = 1
            else:
                dictionary[count] = dictionary[count] + 1
        keys = dictionary.keys()
        values = dictionary.values()
        fig, ax = plt.subplots()
        ax.bar(keys, values)
        plt.xlim([0, 900]) #pamietaj ze niektore teksty siegaja 2000 znaków a przyciety jest dla widocznosci
        plt.show()

    def get_top_n_words(self, n=20, verbose=True):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.loader.get_reviews())
        dictionary = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())
        list_of_words_frequency = (sorted(list(dictionary), key=lambda x: x[1], reverse=True)[:n])
        if verbose:
            for tuple in list_of_words_frequency:
                print(tuple[0] + " " + str(tuple[1]))
        return list_of_words_frequency

    def get_top_20_words_from_each_class(self):
        df = self.loader.get_data()
        df_class_reviews_zip_list = list(zip(df['4class'], df['review']))
        list_of_class_0 = [x for x in df_class_reviews_zip_list if x[0] == 0]
        list_of_class_1 = [x for x in df_class_reviews_zip_list if x[0] == 1]
        list_of_class_2 = [x for x in df_class_reviews_zip_list if x[0] == 2]
        list_of_class_3 = [x for x in df_class_reviews_zip_list if x[0] == 3]
        lists = [list_of_class_0, list_of_class_1, list_of_class_2, list_of_class_3]
        top_500_popular_words = list(map(lambda x: x[0], self.get_top_n_words(n=500, verbose=False)))
        vectorizer = CountVectorizer()
        for index, list_of_class in enumerate(lists):
            print(f"CLASS: {index}")
            X = vectorizer.fit_transform(map(lambda x: x[1], list_of_class))
            dictionary = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())
            list_of_words_frequency = (sorted(list(dictionary), key=lambda x: x[1], reverse=True))
            list_of_words_not_in_top_250 = [el for el in list_of_words_frequency if el[0] not in top_500_popular_words]
            # print(list_of_words_not_in_top_50)
            for tuple in list_of_words_not_in_top_250[:20]:
                print(tuple[0])
