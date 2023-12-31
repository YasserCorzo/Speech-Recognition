import numpy as np

class DiagGauss:

    def __init__(self, feature_size: int):
        """
        DiagGauss is the class of multivariate gaussian distribution with diagonal covariance

        Args:
            feature_size (int): the number of feature (dimension) in gaussian
        """
        self.feature_size = feature_size

        self.mean = np.zeros(self.feature_size)
        self.std = np.ones(self.feature_size)

        # const is the log of constant independent of observation,
        # you might want to compute this number in advance (in set_std) to speed up your logpdf
        self.const = 0.0

        # update std and const
        self.set_std(self.std)

    def set_mean(self, mean: np.ndarray) -> None:
        """
        update the mean of your guassian

        Args:
            mean (np.ndarray): np array with the shape of [D] where D is the feature size

        Returns:
        """

        assert len(mean.shape) == 1, 'mean should have one dimension'
        assert mean.shape[0] == self.feature_size, 'mean dim should match with the feature size'

        self.mean = mean

    def set_std(self, std: np.ndarray) -> None:
        """
        update the standard deviation of your guassian

        Args:
            std (np.ndarray): np array with the shape of [D] where D is the feature size

        Returns:
        """

        assert len(std.shape) == 1, 'std should have one dimension'
        assert std.shape[0] == self.feature_size, 'std dim should match with the feature size'

        self.std = std
        self.const = -0.5*self.feature_size*np.log(2.0*np.pi)-np.sum(np.log(self.std))


    def fit(self, X: np.ndarray) -> None:
        """
        fit the model

        Args:
            X (np.ndarray): X represents your sample matrix. It is two dimension numpy array with [N,D] shape
            where N is the sample size and D is the feature size

        Returns:
        """

        assert len(X.shape) == 2, 'X should have two dim'
        assert X.shape[1] == self.feature_size, 'the second dim should match with your feature size'

        # implement your MLE
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Update the constant term 
        self.const = -0.5 * self.feature_size * np.log(2.0 * np.pi) - np.sum(np.log(self.std))

    def logpdf(self, X: np.ndarray):
        """
        compute the log pdf of your sample

        Args:
            X (np.ndarray): X represents your sample matrix. It is two dimension numpy array with [N,D] shape
            where N is the sample size and D is the feature size

        Returns:
            A 1 dimension numpy array of [N] shape where each element is the logpdf of X_i (i-th row of X)

        """

        assert len(X.shape) == 2, 'X should have two dim'
        assert X.shape[1] == self.feature_size, 'the second dim should match with your feature size'

        non_const = 0.5 * np.sum(np.power((X - self.mean) / self.std, 2), axis=1)
        logpdf = self.const - non_const

        return logpdf