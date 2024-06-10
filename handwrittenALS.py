from tqdm.auto import tqdm
import numpy as np
from sklearn.linear_model import Ridge


class CustomALS(object):
    """
    Класс ALS матричной факторизации.

    Parameters
    ----------
    k : int
        Количество факторов.
    n_iter : int
        Количество итераций подбора матриц U и V.
    lambda_u : float
        Коэффициент регуляризации для матрицы U.
    lambda_v : float
        Коэффициент регуляризации для матрицы V.

    Attributes
    ----------
    U : numpy.ndarray
        Матрица U, размером m*k
    V : numpy.ndarray
        Матрица V, размером n*k
    R_hat : numpy.ndarray
        Аппроксимирующая матрица, размером m*n (R_hat = U*V_transpose)

    """

    def __init__(self, k=1, n_iter=20, lambda_u=0.001, lambda_v=0.001):

        self.k = k
        self.n_iter = n_iter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

    def fit(self, R):
        self.R = R.copy()

        m, n = R.shape

        self.U = np.random.normal(loc=0.0, scale=0.01, size=(m, self.k))
        self.V = np.random.normal(loc=0.0, scale=0.01, size=(n, self.k))

        R_T = self.R.T

        model_u = Ridge(alpha=self.lambda_u, fit_intercept=True)

        model_v = Ridge(alpha=self.lambda_v, fit_intercept=True)

        for _ in tqdm(range(self.n_iter)):

            for i in range(m):
                model_u.fit(X=self.V, y=R_T[:, i])
                self.U[i, :] = model_u.coef_

            for j in range(n):
                model_v.fit(X=self.U, y=R_T[j, :])
                self.V[j, :] = model_v.coef_

        self.R_hat = self.U.dot(self.V.T)
