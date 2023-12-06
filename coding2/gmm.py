import numpy as np
from gauss import DiagGauss
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class DiagGMM:

    def __init__(self, component_size: int, feature_size: int):
        """
        DiagGMM is the class of Gaussian Mixture Model whose components are diagonal Gaussian distributions

        Args:
            component_size (int): the mixture number of Gaussian distribution
            feature_size (int): the number of feature in the diagonal Gaussian distribution
        """
        self.component_size = component_size
        self.feature_size = feature_size

        # mixture weight
        self.log_weight = None

        # all gaussian distributions
        self.gauss = None


    def initialize(self, X: np.ndarray):
        """
        Initialize the GMM model with sample X

        Args:
            X (np.ndarray): the sample matrix, same as fit interface

        Returns:

        """
        # initialize all gauss distribution
        self.gauss = [DiagGauss(self.feature_size) for i in range(self.component_size)]

        # initial weight
        self.log_weight = np.log(np.ones(self.component_size)/self.component_size)

        # you can use kmeans to do the initialize the model
        kmeans_cluster = KMeans(n_clusters=self.component_size)

        sample_size = X.shape[0]

        # initialize with hard assignments
        assignment = kmeans_cluster.fit_predict(X)

        # update each componenet
        for i in range(self.component_size):
            self.log_weight[i] = np.log(np.sum(assignment == i)/sample_size)

            # extract all frame assigned to i
            X_i = X[assignment == i]

            # compute MLE mean and std
            mean = np.mean(X_i, axis=0)
            std = np.sqrt(np.sum(((X_i - mean)**2.0), axis=0)/len(X_i))

            # update mean and std
            self.gauss[i].set_mean(mean)
            self.gauss[i].set_std(std)


    def E_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute the latent responsibilies for each sample and component
        Args:
            X (np.ndarray): [N, D] matrix where N is the sample size and D is the feature size

        Returns:
            a matrix with the size [N, C] where C is the component size and N is sample size.
            The ij-entry is the responsibility of j-th component in i-th sample

        """

        resp_param = np.zeros((X.shape[0], self.component_size))
        
        for k in range(self.component_size):
            resp_param[:, k] = self.log_weight[k] + self.gauss[k].logpdf(X)
        
        resp_param -= logsumexp(resp_param, axis=1).reshape(-1, 1)

        return resp_param

    def M_step(self, X, comp_weight) -> None:
        """
        Maximization step: use the responsibilies (comp_weight) to update your GMM model.
        In particular, you might want to update three parameters:
        - mixture weight (log_weight)
        - mean of each Gaussian component
        - std of each Gaussian component

        Args:
            X (np.ndarray): [N, D] matrix where N is the sample size and D is the feature size
            comp_weight (np.ndarray): [N, C] matrix of component responsibilities. It is the returned value from E step.
            C is the component size and N is le size. The ij-cell is the responsibility of j-th component in i-th sample

        Returns:
        """

        # update mixture weight
        for k in range(len(self.log_weight)):
            self.log_weight[k] = logsumexp(comp_weight[:, k]) - logsumexp(comp_weight)

        # update mean of each Gaussian component
        for k in range(self.component_size):
            #mean = logsumexp(comp_weight[:, k].reshape(-1, 1) + np.log(X), axis=0) - logsumexp(comp_weight[:, k])
            mean = np.sum(np.exp(comp_weight[:, k].reshape(-1, 1)) * X, axis=0) / np.sum(np.exp(comp_weight[:, k]))
            self.gauss[k].set_mean(mean)

        # update std of each Gaussian component
        for k in range(self.component_size):
            std = np.sqrt(np.sum(np.exp(comp_weight[:, k].reshape(-1, 1)) * np.power(X - self.gauss[k].mean, 2), axis=0) / np.sum(np.exp(comp_weight[:, k])))
            self.gauss[k].set_std(std)

    def fit(self, X: np.ndarray):
        """
        fit the GMM model with your sample X.
        You should update your model iteratively with EM algorithm

        Args:
            X (np.ndarray): sample matrix of shape [N, D] where N is the number of sample (frame),
            D is the feature size

        Returns:
        """

        # estimate the GMM with kmeans
        if self.gauss is None:
            self.initialize(X)

        # EM steps
        for i in range(40):

            # compute the responsibility
            comp_weight = self.E_step(X)

            # compute
            self.M_step(X, comp_weight)


    def logpdf(self, X: np.ndarray):
        """
        compute the GMM logpdf of a sample

        Args:
            X (np.ndarray): sample matrix of shape [N, D] where N is the number of sample (frame)
            D is the feature size

        Returns:
            an np array of shape [N] where each element is the logpdf of X_i (the i-th row in X)
        """

        logprob_lst = []
        for i in range(self.component_size):
            logprob_lst.append(self.gauss[i].logpdf(X) + self.log_weight[i])

        # sum probability
        logprob = logsumexp(logprob_lst, axis=0)
        return logprob
