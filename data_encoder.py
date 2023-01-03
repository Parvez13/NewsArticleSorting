from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# One hot encoder

def onehotencoder(y_train, y_test):
    """
    Args:
        y_train: Training labels
        y_test : Testing labels
    return:
       Returns 'y_train' and 'y_test' labels as one-hot encode(1,0,0,0,0)
    """
    try:
        # Initialize oneHotEncoder to our labels
        one_hot_encoder = OneHotEncoder(sparse=False)
        # Train labels
        y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
        # Test labels
        y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))
        return y_train_one_hot, y_test_one_hot
    except Exception as e:
        print(f'Error raised!!: {e}')

# Label Encoder
def labelencoder(y_train, y_test):
    """
    Args:
        y_train: Training labels
        y_test: Testing labels
    returns:
        Returns 'y_train' and 'y_test' labels as int-encoded(1, 2, 3, 4, 5)

    """
    try:
        # Intialize label encoder to our labels
        label_encoder = LabelEncoder()
        # Train labels
        y_train_labels_encoded = label_encoder.fit_transform(y_train.reshape(-1, 1))
        # Test labels
        y_test_labels_encoded = label_encoder.transform(y_test.reshape(-1, 1))
        # Class names
        class_names = label_encoder.classes_
        return y_train_labels_encoded, y_test_labels_encoded, class_names
    except Exception as e:
        print(f'Error raised!!: {e}')

