from DataAnalyzer import DataAnalyzer
from Loader import Loader

if __name__ == '__main__':
    loader = Loader()
    dataAnalyzer = DataAnalyzer(loader)
    # dataAnalyzer.get_4class_frequency_occurrence()
    # dataAnalyzer.get_review_length()
    # dataAnalyzer.get_top_n_words()
    dataAnalyzer.get_top_20_words_from_each_class()