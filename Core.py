from Loader import Loader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC




class Core:

    def __init__(self):
        self.loader = Loader()
        self.labels = None
        self.features = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.prepare_data()

    def prepare_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.loader.get_reviews(), self.loader.get_4class())

    def bayes_analysis(self, alpha=1.0, fit_prior=True, class_prior=None):
        # create naive bayes pipeline with vectorizer
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(sublinear_tf=True, min_df=0, max_features=5000, norm='l2', stop_words='english')),
            ('classifier', MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior))
        ])
        #fit models into the pipeline
        pipeline.fit(self.x_train, self.y_train)

        return pipeline

    def find_best_parameters_for_byes(self, pipeline):
        parameters = {
            'vectorizer__max_features': [None, 1_000, 5_000, 10_000],
            'vectorizer__min_df': [0, 0.05, 0.1, 0.2, 0.5, 1],
            'classifier__alpha': [1, 0.1, 0.01, 0.0001, 0],
            'classifier__fit_prior': [True, False]
        }
        grid_search = GridSearchCV(pipeline, parameters, cv=10, n_jobs=4, verbose=3)
        grid_search.fit(self.x_train, self.y_train)
        print(f"Best score {grid_search.best_score_}")
        best_params = grid_search.best_estimator_.get_params()
        print("Best parameters:")
        for param in sorted(parameters.keys()):
            print(f"{param}: {best_params[param]}")

    def predict_bayes(self, alpha=1.0, fit_prior=True, class_prior=None):
        pipeline = self.bayes_analysis(alpha, fit_prior, class_prior)
        return pipeline.score(self.x_test, self.y_test)

    def svc_analysis(self, C=1.0, kernel='rbf', gamma='scale'):
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(sublinear_tf=True, min_df=5, max_features=5000, norm='l2', stop_words='english')),
            ('classifier', SVC(C=C, kernel=kernel, gamma=gamma))
        ])
        pipeline.fit(self.x_train, self.y_train)
        return pipeline

    def find_best_parameters_for_svc(self, pipeline):
        parameters = {
            'classifier__C': [1, 10, 100, 1000],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['auto', 'scale']
        }
        grid_search = GridSearchCV(pipeline, parameters, cv=10, n_jobs=3, verbose=3)
        grid_search.fit(self.x_train, self.y_train)
        print(f"Best score {grid_search.best_score_}")
        best_params = grid_search.best_estimator_.get_params()
        print("Best parameters:")
        for param in sorted(parameters.keys()):
            print(f"{param}: {best_params[param]}")

    def predict_svc(self, C=1.0, kernel='rbf', gamma='scale'):
        pipeline = self.svc_analysis(C=C, kernel=kernel, gamma=gamma)
        return pipeline.score(self.x_test, self.y_test)

