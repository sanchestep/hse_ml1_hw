from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample
        self.num_of_features: int

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):

        idxs = np.random.choice(range(x.shape[0]), size=int(self.subsample * x.shape[0]))
        x_bootstrap = x[idxs]
        y_bootstrap = y[idxs]
        pred_bootstrap = predictions[idxs]

        base_model = self.base_model_class(**self.base_model_params)
        grad = self.loss_derivative(y_bootstrap, pred_bootstrap)
        base_model.fit(x_bootstrap, -grad)

        base_model_preds = base_model.predict(x_bootstrap)

        optimal_gamma = self.find_optimal_gamma(y_bootstrap, pred_bootstrap, base_model_preds)

        predictions[idxs] += self.learning_rate * optimal_gamma * base_model_preds

        self.gammas.append(self.learning_rate * optimal_gamma)
        self.models.append(base_model)


    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            if self.early_stopping_rounds is not None:
              train_loss = self.loss_fn(y_train, train_predictions)
              valid_loss = self.loss_fn(y_valid, valid_predictions)

              self.history['train_loss'].append(train_loss)
              self.history['valid_loss'].append(valid_loss)
              
              if valid_loss < np.min(self.validation_loss):
                self.validation_loss = np.append(self.validation_loss[1:], valid_loss)
              else:
                break
            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)


        if self.plot:
          plt.plot(self.history['train_loss'], self.models)
          plt.plot(self.history['test_loss'], self.models)

    def predict_proba(self, x):
        res = np.zeros((x.shape[0], 2))
        for gamma, model in zip(self.gammas, self.models):
          res[:, 1] += gamma * model.predict(x)

        return self.sigmoid(res)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    # @property
    def feature_importances_(self, x):
        total_importances = np.zeros(x.shape[1])

        for model in self.models:
          total_importances += model.feature_importances_

        total_importances /= self.n_estimators

        return total_importances
