from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def split_train_val_test(data_set, validation_size = 0.10, test_size = 0.20, random_state = 42):
    assert len(data_set) != 0
    assert (validation_size + test_size) < 1.0
    temp_set, test_set = train_test_split(data_set, test_size=test_size, random_state=random_state)
    if validation_size == 0:
        return temp_set, [], test_set
    train_set, validation_set = train_test_split(temp_set, test_size=validation_size/(1.0-test_size), random_state=random_state)
    return train_set, validation_set, test_set

def plot_roc(ground_truth_labels, prediction_labels, title: str):
    RocCurveDisplay.from_predictions(
        ground_truth_labels,
        prediction_labels,
        name=f"class_of_interest vs the rest",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    # plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()
