from sklearn.dummy import DummyRegressor

def train_dummy_regressor(X_train, y_train, strategy='mean'):
    """
    Entrena y devuelve un modelo DummyRegressor simple.
    """
    model = DummyRegressor(strategy=strategy)
    model.fit(X_train, y_train)
    return model
