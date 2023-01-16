from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, f1_score
import matplotlib.pyplot as plt

def split_train_val_test(data_set, validation_size = 0.10, test_size = 0.20, random_state = 42):
    """
    Splits the given dataset into a train, a validation and a test set using the random state for shuffling before splitting.
    """
    assert len(data_set) != 0
    assert (validation_size + test_size) < 1.0
    temp_set, test_set = train_test_split(data_set, test_size=test_size, random_state=random_state)
    if validation_size == 0:
        return temp_set, [], test_set
    train_set, validation_set = train_test_split(temp_set, test_size=validation_size/(1.0-test_size), random_state=random_state)
    return train_set, validation_set, test_set

def plot_roc(model, X, y, title: str):
    """
    Plots the Receiver Operating Characteristic with a reference line of a dummy classifier.
    """
    rocDisp = RocCurveDisplay.from_estimator(model, X, y, color="orange")
    plt.plot([0, 1], [0, 1], "k--", label="Reference (AUC = 0.5)")
    # plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()


def calc_f1(model, X, y_true, average='weighted'):
    y_pred = model.predict(X)
    return f1_score(y_true, y_pred, average=average)