from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Utilidades de MAPIE v1 (ya las tienes importadas en tu archivo)
from mapie.utils import (
    _transform_confidence_level_to_alpha_list,
    _prepare_params,
    _check_estimator_fit_predict,
    _raise_error_if_previous_method_not_called,
    _raise_error_if_method_already_called,
    _raise_error_if_fit_called_in_prefit_mode,
)
from mapie.utils import _cast_point_predictions_to_ndarray, _cast_predictions_to_ndarray_tuple


class MinMaxIntervalRegressor:
    """
    Regresor de intervalos tipo 'baseline' que usa el mínimo y máximo observados
    en el conjunto de conformalización para construir intervalos constantes
    [y_min, y_max], independientemente de X.

    Flujo:
      1) fit(X_train, y_train)        -> opcional si prefit=True
      2) conformalize(X_cal, y_cal)   -> calcula y_min_ y y_max_
      3) predict_interval(X)          -> devuelve (y_pred, y_pis) con formato MAPIE

    Parámetros
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        Estimador base para predicciones puntuales.

    confidence_level : Union[float, Iterable[float]], default=0.9
        Niveles de confianza solicitados. Se usan solo para la forma de salida:
        el intervalo [y_min, y_max] se repite para cada nivel.

    prefit : bool, default=True
        Si True, se asume que el estimator ya viene ajustado y se omite fit().
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        prefit: bool = True
    ) -> None:
        _check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False

        # Guardamos niveles (solo para definir la tercera dimensión de salida)
        self._alphas = _transform_confidence_level_to_alpha_list(confidence_level)
        self._predict_params: dict = {}

        # Atributos que se rellenan en conformalize
        self.y_min_: Optional[float] = None
        self.y_max_: Optional[float] = None

    # ----------------------------- PUBLIC API ----------------------------- #

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> "MinMaxIntervalRegressor":
        """
        Ajusta el estimator si prefit=False. Si prefit=True, lanza error si se llama.
        """
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)

        cloned = clone(self._estimator)
        fit_params_ = _prepare_params(fit_params)
        cloned.fit(X_train, y_train, **fit_params_)
        self._estimator = cloned

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_cal: ArrayLike,
        y_cal: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> "MinMaxIntervalRegressor":
        """
        Calcula y_min_ y y_max_ a partir de y_cal. Guarda predict_params
        para usarlos en predict/predict_interval.
        """
        _raise_error_if_previous_method_not_called(
            "conformalize", "fit", self._is_fitted
        )
        _raise_error_if_method_already_called(
            "conformalize", self._is_conformalized
        )

        y_cal = np.asarray(y_cal)
        if y_cal.size == 0:
            raise ValueError("y_cal no puede estar vacío para MinMaxIntervalRegressor.")

        self.y_min_ = float(np.min(y_cal))
        self.y_max_ = float(np.max(y_cal))

        self._predict_params = _prepare_params(predict_params)
        self._is_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,  # compat
        allow_infinite_bounds: bool = False,     # compat
    ) -> Tuple[NDArray, NDArray]:
        """
        Devuelve:
          - y_pred: (n_samples,)
          - y_pis:  (n_samples, 2, n_confidence_levels)
                    [:,0,:] = y_min_, [:,1,:] = y_max_ (constantes)
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval", "conformalize", self._is_conformalized
        )
        if self.y_min_ is None or self.y_max_ is None:
            raise RuntimeError("Debes llamar a conformalize() antes de predecir intervalos.")

        # Predicciones puntuales del estimator
        y_pred = self._estimator.predict(X, **self._predict_params)
        y_pred = _cast_point_predictions_to_ndarray(y_pred)  # forma (n_samples,)

        n = y_pred.shape[0]
        k = len(self._alphas)

        # Intervalos constantes replicados para cada muestra y cada nivel
        lower = np.full((n, 1, k), self.y_min_, dtype=float)
        upper = np.full((n, 1, k), self.y_max_, dtype=float)
        y_pis = np.concatenate([lower, upper], axis=1)  # (n, 2, k)

        # Mantener coherencia con utilidades de casteo (aunque aquí ya es ndarray)
        return _cast_predictions_to_ndarray_tuple((y_pred, y_pis))

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Solo predicciones puntuales (requiere conformalize previo, igual que en v1).
        """
        _raise_error_if_previous_method_not_called(
            "predict", "conformalize", self._is_conformalized
        )
        y_pred = self._estimator.predict(X, **self._predict_params)
        return _cast_point_predictions_to_ndarray(y_pred)
    

class GaussianHomoskedasticPIRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        prefit: bool = True,
    ) -> None:
        _check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False
        self._alphas = _transform_confidence_level_to_alpha_list(confidence_level)
        self._predict_params: dict = {}
        self.sigma_: Optional[float] = None

    def fit(self, X: ArrayLike, y: ArrayLike, fit_params: Optional[dict] = None):
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)
        est = clone(self._estimator)
        est.fit(X, y, **_prepare_params(fit_params))
        self._estimator = est
        self._is_fitted = True
        return self

    def conformalize(self, X_cal: ArrayLike, y_cal: ArrayLike, predict_params: Optional[dict] = None):
        _raise_error_if_previous_method_not_called("conformalize", "fit", self._is_fitted)
        _raise_error_if_method_already_called("conformalize", self._is_conformalized)
        self._predict_params = _prepare_params(predict_params)
        y_hat = self._estimator.predict(X_cal, **self._predict_params)
        resid = np.asarray(y_cal) - np.asarray(y_hat)
        self.sigma_ = float(np.std(resid, ddof=1))
        self._is_conformalized = True
        return self

    def predict_interval(self, X: ArrayLike) -> Tuple[NDArray, NDArray]:
        _raise_error_if_previous_method_not_called("predict_interval", "conformalize", self._is_conformalized)
        assert self.sigma_ is not None
        y_pred = _cast_point_predictions_to_ndarray(self._estimator.predict(X, **self._predict_params))
        n, k = y_pred.shape[0], len(self._alphas)
        z = [norm.ppf(1 - a/2) for a in self._alphas]
        lows = np.stack([y_pred - zz*self.sigma_ for zz in z], axis=1)   # (n,k)
        ups  = np.stack([y_pred + zz*self.sigma_ for zz in z], axis=1)
        y_pis = np.stack([lows, ups], axis=1)  # (n,2,k)
        return _cast_predictions_to_ndarray_tuple((y_pred, y_pis))

    def predict(self, X: ArrayLike) -> NDArray:
        _raise_error_if_previous_method_not_called("predict", "conformalize", self._is_conformalized)
        return _cast_point_predictions_to_ndarray(self._estimator.predict(X, **self._predict_params))


class ResidualBootstrapPIRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        prefit: bool = True,
        n_boot: int = 1000,
        random_state: Optional[int] = 42,
        use_empirical: bool = False,   # ← NEW: usar cuantiles empíricos (sin bootstrap)
    ) -> None:
        _check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False
        self._alphas = _transform_confidence_level_to_alpha_list(confidence_level)
        self._predict_params: dict = {}
        self.n_boot = int(n_boot)
        self.rng = np.random.default_rng(random_state)
        self.use_empirical = use_empirical
        self.residuals_: Optional[NDArray] = None
        self._q_lo_: Optional[NDArray] = None  # guardamos si use_empirical=True
        self._q_hi_: Optional[NDArray] = None

    @staticmethod
    def _to_2d(X):
        X = np.asarray(X)
        return X if X.ndim == 2 else X.reshape(-1, 1)

    def fit(self, X, y, fit_params=None):
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)
        est = clone(self._estimator)
        est.fit(self._to_2d(X), np.asarray(y), **_prepare_params(fit_params))
        self._estimator = est
        self._is_fitted = True
        return self

    def conformalize(self, X_cal, y_cal, predict_params=None):
        _raise_error_if_previous_method_not_called("conformalize", "fit", self._is_fitted)
        _raise_error_if_method_already_called("conformalize", self._is_conformalized)

        self._predict_params = _prepare_params(predict_params)
        y_hat = self._estimator.predict(self._to_2d(X_cal), **self._predict_params)
        r = np.asarray(y_cal) - np.asarray(y_hat)

        # limpieza por si hay NaN/inf
        r = r[np.isfinite(r)]
        if r.size == 0:
            raise ValueError("Empty/invalid residuals for bootstrap.")
        self.residuals_ = r

        if self.use_empirical:
            # determinista y barato: cuantiles directos de residuales
            alphas = np.array(list(self._alphas), dtype=float)
            self._q_lo_ = np.quantile(r, alphas/2.0)
            self._q_hi_ = np.quantile(r, 1.0 - alphas/2.0)

        self._is_conformalized = True
        return self

    def predict_interval(self, X):
        _raise_error_if_previous_method_not_called("predict_interval", "conformalize", self._is_conformalized)
        y_pred = _cast_point_predictions_to_ndarray(self._estimator.predict(self._to_2d(X), **self._predict_params))
        n = y_pred.shape[0]

        if self.use_empirical:
            # ancho constante; no Monte-Carlo
            lows = y_pred[:, None] + self._q_lo_[None, :]
            ups  = y_pred[:, None] + self._q_hi_[None, :]
        else:
            if self.n_boot <= 0:
                raise ValueError("n_boot must be > 0 for bootstrap mode.")
            # muestrea residuales i.i.d. (matriz n x B) y añade a y_pred
            R = self.rng.choice(self.residuals_, size=(n, self.n_boot), replace=True)
            sims = y_pred[:, None] + R
            alphas = np.array(list(self._alphas), dtype=float)
            lows = np.quantile(sims, alphas/2.0, axis=1).T  # (n, k)
            ups  = np.quantile(sims, 1.0 - alphas/2.0, axis=1).T

        y_pis = np.stack([lows, ups], axis=1)  # (n,2,k)
        return _cast_predictions_to_ndarray_tuple((y_pred, y_pis))



from scipy.stats import poisson, nbinom

class PoissonIntervalRegressor:
    """
    Mean-to-quantiles interval regressor for counts.
    Same API as your other generators.

    - If use_negbin=False → Poisson intervals.
    - If use_negbin=True  → Negative Binomial intervals with Var = mu + alpha*mu^2,
      where alpha is estimated on the calibration set (method of moments) unless
      alpha_overdisp is provided.

    Parameters
    ----------
    estimator : RegressorMixin
        Any model that predicts the mean mu(x) (>=0). With no covariates,
        pass your mean model (e.g., naive_model).
    confidence_level : float | Iterable[float]
    prefit : bool
        If True, `estimator` comes already fitted.
    use_negbin : bool
    alpha_overdisp : Optional[float]
        Fixed over-dispersion; if None and use_negbin=True, it is estimated in `conformalize`.
    min_mu : float
        Lower bound to keep mu>0.
    integerize_bounds : bool
        If True, return floor/ceil integer bounds.
    """
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        prefit: bool = True,
        use_negbin: bool = False,
        alpha_overdisp: Optional[float] = None,
        min_mu: float = 1e-9,
        integerize_bounds: bool = True,
    ) -> None:
        _check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False
        self._alphas = _transform_confidence_level_to_alpha_list(confidence_level)
        self._predict_params: dict = {}

        self.use_negbin = bool(use_negbin)
        self.alpha_overdisp = alpha_overdisp
        self.alpha_: Optional[float] = None
        self.min_mu = float(min_mu)
        self.integerize_bounds = bool(integerize_bounds)

    @staticmethod
    def _clip_mu(mu: ArrayLike, min_mu: float) -> NDArray:
        mu = np.asarray(mu, dtype=float)
        return np.maximum(mu, min_mu)

    def fit(self, X: ArrayLike, y: ArrayLike, fit_params: Optional[dict] = None):
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)
        est = clone(self._estimator)
        est.fit(X, y, **_prepare_params(fit_params))
        self._estimator = est
        self._is_fitted = True
        return self

    def conformalize(self, X_cal: ArrayLike, y_cal: ArrayLike, predict_params: Optional[dict] = None):
        _raise_error_if_previous_method_not_called("conformalize", "fit", self._is_fitted)
        _raise_error_if_method_already_called("conformalize", self._is_conformalized)

        self._predict_params = _prepare_params(predict_params)

        if self.use_negbin and (self.alpha_overdisp is None):
            # Method-of-moments for over-dispersion on calibration set:
            y_cal = np.asarray(y_cal, dtype=float)
            mu_cal = self._clip_mu(self._estimator.predict(X_cal, **self._predict_params), self.min_mu)
            s2 = float(np.var(y_cal, ddof=1)) if y_cal.size > 1 else 0.0
            m1 = float(np.mean(mu_cal))
            m2 = float(np.mean(mu_cal**2))
            alpha = 0.0 if m2 == 0 else (s2 - m1) / m2
            self.alpha_ = float(max(0.0, alpha))
        else:
            # Poisson (alpha=0) or user-specified alpha
            self.alpha_ = float(self.alpha_overdisp or 0.0)

        self._is_conformalized = True
        return self

    def _ppf_bounds(self, mu: NDArray, a: float) -> Tuple[NDArray, NDArray]:
        if self.use_negbin and (self.alpha_ or 0.0) > 0:
            # NB parameterization: r = 1/alpha, p = r/(r+mu)
            r = 1.0 / self.alpha_
            p = r / (r + mu)
            lo = nbinom.ppf(a/2.0, r, p)
            hi = nbinom.ppf(1.0 - a/2.0, r, p)
        else:
            lo = poisson.ppf(a/2.0, mu)
            hi = poisson.ppf(1.0 - a/2.0, mu)

        if self.integerize_bounds:
            lo = np.floor(lo)
            hi = np.ceil(hi)
        return lo.astype(float), hi.astype(float)

    def predict_interval(self, X: ArrayLike) -> Tuple[NDArray, NDArray]:
        _raise_error_if_previous_method_not_called("predict_interval", "conformalize", self._is_conformalized)
        mu = self._clip_mu(self._estimator.predict(X, **self._predict_params), self.min_mu)
        y_pred = _cast_point_predictions_to_ndarray(mu)  # point = mean

        lows, ups = [], []
        for a in self._alphas:
            lo, hi = self._ppf_bounds(mu, float(a))
            lows.append(lo)
            ups.append(hi)
        lows = np.stack(lows, axis=1)  # (n, k)
        ups  = np.stack(ups,  axis=1)

        y_pis = np.stack([lows, ups], axis=1)  # (n, 2, k)
        return _cast_predictions_to_ndarray_tuple((y_pred, y_pis))

    def predict(self, X: ArrayLike) -> NDArray:
        _raise_error_if_previous_method_not_called("predict", "conformalize", self._is_conformalized)
        mu = self._clip_mu(self._estimator.predict(X, **self._predict_params), self.min_mu)
        return _cast_point_predictions_to_ndarray(mu)
