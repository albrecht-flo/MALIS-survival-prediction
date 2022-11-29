from sklearn.model_selection import train_test_split

def split_train_val_test(data_set, validation_size = 0.10, test_size = 0.20, random_state = 42):
    assert len(data_set) != 0
    assert (validation_size + test_size) < 1.0
    temp_set, test_set = train_test_split(data_set, test_size=test_size, random_state=random_state)
    if validation_size == 0:
        return temp_set, [], test_set
    train_set, validation_set = train_test_split(temp_set, test_size=validation_size/(1.0-test_size), random_state=random_state)
    return train_set, validation_set, test_set