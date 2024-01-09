import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    sort_idxs = np.argsort(feature_vector)
    feature_vector = feature_vector[sort_idxs]
    target_vector = target_vector[sort_idxs]

    feature_vector, idx = np.unique(feature_vector, return_index=True)
    thresholds = (feature_vector[:-1] + feature_vector[1:]) / 2
    idx = idx[1:len(feature_vector)] - 1
    pref = np.cumsum(target_vector)

    R_l = idx + 1
    R_r = len(target_vector) - R_l
    num1_l = pref[idx]
    num1_r = pref[-1] - num1_l
    num0_l = R_l - num1_l
    num0_r = R_r - num1_r
    ginis = -((R_l / len(target_vector)) * (1 - (num1_l / R_l)**2 - (num0_l / R_l)**2) + (R_r / len(target_vector)) * (1 - (num1_r / R_r)**2 - (num0_r / R_r)**2))

    best_gini = ginis[np.argmax(ginis)]
    best_thershold = thresholds[np.argmax(ginis)]

    return thresholds, ginis, best_thershold, best_gini


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
      if len(sub_y) == 0 or (np.all(sub_y != sub_y[0]) and len(sub_y) < self._min_samples_split):
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]
        return

      feature_best, threshold_best, gini_best, split = None, None, None, None

      for feature in range(sub_X.shape[1]):
        feature_type = self._feature_types[feature]
        categories_map = {}

        if feature_type == "real":
            feature_vector = sub_X[:, feature]
        elif feature_type == "categorical":
            counts = Counter(sub_X[:, feature])
            clicks = Counter(sub_X[sub_y == 1, feature])
            ratio = {key: clicks[key] / count if key in clicks else 0 for key, count in counts.items()}
            sorted_categories = sorted(ratio, key=ratio.get)
            categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))

            feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
        else:
            raise ValueError

        if len(np.unique(feature_vector)) == 1:
            continue

        _, _, threshold, gini = find_best_split(feature_vector, sub_y)

        if (gini_best is None or gini > gini_best) and (gini is not None):
            feature_best = feature
            gini_best = gini
            split = feature_vector < threshold

            if feature_type == "real":
                threshold_best = threshold
            elif feature_type == "categorical":
                threshold_best = [key for key, val in categories_map.items() if val < threshold]
            else:
                raise ValueError

      if feature_best is None:
        node["type"] = "terminal"
        node["class"] = Counter(sub_y).most_common(1)[0][0]
        return

      node["type"] = "nonterminal"
      node["feature_split"] = feature_best

      if self._feature_types[feature_best] == "real":
          node["threshold"] = threshold_best
      elif self._feature_types[feature_best] == "categorical":
          node["categories_split"] = threshold_best
      else:
          raise ValueError

      node["left_child"], node["right_child"] = {}, {}
      self._fit_node(sub_X[split], sub_y[split], node["left_child"])
      self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node['type'] == 'terminal':
          return node['class']
        feature = node['feature_split']
        next_node = {}

        if self._feature_types[feature] == 'real':
          if x[feature] < node['threshold']:
            next_node = node['left_child']
          else:
            next_node = node['right_child']
        elif self._feature_types[feature] == 'categorical':
          if x[feature] in node['categorical_split']:
            next_node = node['left_child']
          else:
            next_node = node['right_child']
        else:
          raise ValueError
        
        return self._predict_node(x, next_node)

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
