import numpy as np
import pytest

from decision_tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseDecisionTree()


@pytest.mark.parametrize("cls", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_hyperparameters(cls):
    model = cls(max_depth=3, min_samples_split=4)
    assert model.get_params()["max_depth"] == 3
    model.set_params(criterion="entropy")
    assert model.criterion == "entropy"
    with pytest.raises(ValueError):
        model.set_params(foo=1)


def test_classifier_fit_predict_eval():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2, criterion="gini")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert model.eval(X, y) == 1.0


def test_classifier_one_hot():
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(criterion="gini")
    model.fit(X, y)
    assert model.eval(X, y) == 1.0


def test_classifier_entropy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)
    assert model.eval(X, y) == 1.0


def test_regressor_fit_predict_eval():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    model = DecisionTreeRegressor(max_depth=2, criterion="mse")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert model.eval(X, y) == pytest.approx(1.0)


def test_max_depth_zero():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(preds == preds[0])
