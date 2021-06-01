from sklearn.svm import SVC

from Core import Core
from DataAnalyzer import DataAnalyzer
from Loader import Loader

def analyzeData():
    loader = Loader()
    dataAnalyzer = DataAnalyzer(loader)
    dataAnalyzer.get_4class_frequency_occurrence()
    dataAnalyzer.get_review_length()
    dataAnalyzer.get_top_n_words()
    dataAnalyzer.get_top_20_words_from_each_class()

if __name__ == '__main__':
    core = Core()
    # core.find_best_parameters_for_byes(core.bayes_analysis())
    # core.find_best_parameters_for_svc(core.svc_analysis())
    # print("BAYES WITH NO NORM AND STOP_WORDS DEFAULT PARAMETERS SCORE:")
    # print(core.predict_bayes())
    # print("BAYES WITH NO NORM AND STOP_WORDS  BEST PARAMETERS SCORE:")
    # print(core.predict_bayes(alpha=0.01, fit_prior=False))
    print("SVM WITH NO NORM AND STOP_WORDS DEFAULT PARAMETERS SCORE:")
    print(core.predict_svc())
    print("SVM WITH NO NORM AND STOP_WORDS BEST PARAMETERS SCORE:")
    print(core.predict_svc(C=1, gamma='auto', kernel='linear'))

