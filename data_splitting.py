from sklearn.model_selection import train_test_split

def data_splitting(X, y):
    """
    Args:
        X : Takes preprocessed sentences as independent features
        y: Takes Category column

    returns X_train, X_test, y_train, y_test
    """
    # Split the 'Independent' and 'Dependent' values as train_test_split
    X_train_sent, X_test_sent, y_train, y_test = train_test_split(X,
                                                                  y,
                                                                  test_size=0.20,
                                                                  random_state=42)
    return X_train_sent, X_test_sent, y_train, y_test