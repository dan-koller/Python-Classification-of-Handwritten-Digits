import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def fit_predict_model(model: object,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> tuple:
    # Fit the model and return the accuracy score
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return model.score(X_test, y_test), accuracy


def print_fitted_model(models: list, res: list):
    # Print the accuracy score for each model
    for i, model in enumerate(models):
        print(
            f"Model: {model.__class__.__name__}\nAccuracy: {res[i][1]: .3f}\n")


def print_best_models(models: list, res: list):
    # Find the two best models and print their accuracy score
    best = sorted(res, key=lambda x: x[0], reverse=True)[:2]
    print(f"The two best models are: {models[res.index(best[0])].__class__.__name__}-{best[0][1]: .3f}, "
          f"{models[res.index(best[1])].__class__.__name__}-{best[1][1]: .3f}")


def fit_predict_grid_search(model: object,
                            features_train: pd.DataFrame,
                            labels_train: pd.Series,
                            features_test: pd.DataFrame,
                            labels_test: pd.Series,
                            params: dict) -> tuple:
    # Fit the model and return the accuracy score
    grid_search = GridSearchCV(model, params, scoring="accuracy", n_jobs=-1)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.best_estimator_.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    return grid_search.score(features_test, labels_test), accuracy


def print_best_model_estimator(x_train_norm: pd.DataFrame,
                               y_train: pd.Series,
                               x_test_norm: pd.DataFrame,
                               y_test: pd.Series):
    # Compare K-nearest neighbors and Random forest algorithms using GridSearchCV
    models = {
        "KNeighborsClassifier": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 4],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "brute"]
            },
            "name": "K-nearest neighbors"
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [300, 500],
                "max_features": ["auto", "log2"],
                "class_weight": ["balanced", "balanced_subsample"],
                "random_state": [40]
            },
            "name": "Random forest"
        }
    }
    res = [fit_predict_grid_search(model["model"], x_train_norm, y_train, x_test_norm, y_test, model["params"]) for
           _, model in models.items()]
    for i, model in enumerate(models):
        print(f"""
            {models[model]["name"]} algorithm
            best estimator: {models[model]["model"]}
            Accuracy: {res[i][1]: .3f}
            """)


class DataAnalysis:
    def __init__(self):
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        n, h, w = X_train.shape
        self.X_train = X_train.reshape(n, h * w)
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Normalized data
        self.x_train_norm = None
        self.x_test_norm = None

    def split_data(self, test_size: float = 0.3, random_state: int = 40):
        # Split data into training and test sets using only 6000 samples
        return train_test_split(self.X_train[:6000], self.y_train[:6000], test_size=test_size,
                                random_state=random_state)

    def get_data(self):
        # Get the (raw) data and return it in the required format
        X_train, X_test, y_train, y_test = self.split_data()
        return f"""
        x_train shape: {X_train.shape}
        x_test shape: {X_test.shape}
        y_train shape: {y_train.shape}
        y_test shape: {y_test.shape}
        Proportion of classes in the training set:
        {pd.Series(y_train).value_counts(normalize=True)}
        """

    def normalize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        # Normalize the data
        normalizer = Normalizer()
        self.x_train_norm = normalizer.fit_transform(X_train)
        self.x_test_norm = normalizer.transform(X_test)
        return self.x_train_norm, self.x_test_norm


def main():
    models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
              LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]

    analysis = DataAnalysis()
    X_train, X_test, y_train, y_test = analysis.split_data()
    x_train_norm, x_test_norm = analysis.normalize_data(X_train, X_test)

    # Fit and predict the models
    # res = [fit_predict_model(model, x_train_norm, y_train, x_test_norm, y_test) for model in models]
    # print_fitted_model(models, res)

    # Find the two best models
    # print_best_models(models, res)

    # Compare K-nearest neighbors and Random forest algorithms and print the best estimator
    print_best_model_estimator(x_train_norm, y_train, x_test_norm, y_test)


if __name__ == "__main__":
    main()
