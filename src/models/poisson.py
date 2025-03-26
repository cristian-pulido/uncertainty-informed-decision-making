from sklearn.linear_model import PoissonRegressor

def train_poisson_regressor(X_train, y_train, alpha=1e-6, max_iter=1000):
    """
    Entrena y devuelve un modelo PoissonRegressor.
    """
    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model
