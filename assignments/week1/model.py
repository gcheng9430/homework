import numpy as np


class LinearRegression:
    """
    a linear regression model that uses closed form formula to make predictions
    """

    # w: np.ndarray
    # b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = None
        self.b = None

    def fit(self, X, y) -> None:
        """
        fit model with given data
        Arguments:
            X(np.ndarray):Input data
            y(np.ndarray): Input label
        Returns:
            None
        """
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = y.mean() - self.w @ X.mean(axis=0)
        # raise NotImplementedError()

    def predict(self, X) -> np.ndarray:
        """
        predict the model output for the given input.
        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input label
        Returns:
            np.ndarray: the prediction
        """
        return X.dot(self.w) + self.b
        # raise NotImplementedError()


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
        """
        fit the model with given data
        Arguments:
        X(np.ndarray): input data
        y(np.ndarray): input label
        lr(float): the learning rate
        epochs(int): the number of iterations
        Returns:
            None
        """
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        for i in range(epochs):
            y_pred = self.predict(X)
            # calculate gradient

            d_w = -(2 * (X.T) @ (y - y_pred)) / self.m
            d_b = -2 * np.sum(y - y_pred) / self.m

            # update params
            self.w -= lr * d_w
            self.b -= lr * d_b

    def predict(self, X: np.ndarray):
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
