import pandas as pd


class Loader:

    def __init__(self):
        self.csv_file = "data/data.csv"
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.csv_file, index_col=0)

    def get_data(self):
        return self.df

    def get_ids(self):
        return self.df["id"]

    def get_3class(self):
        return self.df["3class"]

    def get_4class(self):
        return self.df["4class"]

    def get_ratings(self):
        return self.df["rating"]

    def get_reviews(self):
        return self.df["review"]

    def get_ids_values(self):
        return self.get_ids().values

    def get_3class_values(self):
        return self.get_3class().values

    def get_4class_values(self):
        return self.get_4class().values

    def get_ratings_values(self):
        return self.get_ratings().values

    def get_reviews_values(self):
        return self.get_reviews().values

