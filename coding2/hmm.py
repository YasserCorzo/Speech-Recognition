import numpy as np
from gmm import DiagGMM
from scipy.special import logsumexp


class HMM:

    def __init__(self, state_size: int, component_size: int, feature_size: int):
        """
        HMM is the acoustic model

        Args:
            state_size (int): the number of state in HMM
            component_size (int): the number of Gaussian in each GMM
            feature_size (int): the number of feature size in each Gaussian
        """

        self.state_size = state_size
        self.component_size = component_size
        self.feature_size = feature_size

        # For each state in HMM, we would like to remember all frames that aligned to them
        self.state2frames = [[] for i in range(self.state_size)]

        # Similarly, for each state, we would like to remember its recursion and forward occurance in all alignments.
        self.state2count = [[0, 0] for i in range(self.state_size)]

        # initialize transition probability
        self.trans_logpdf = np.zeros((self.state_size, self.state_size))

        # create your GMM
        self.gmms = []
        for i in range(self.state_size):
            self.gmms.append(DiagGMM(self.component_size, self.feature_size))

    def initialize(self):
        """
        Initialize your HMM and GMM

        Returns:
        """

        # initialize transition model
        self.trans_logpdf = np.zeros((self.state_size, self.state_size))

        # we follow Kaldi's style by setting recursive/forward prob to 0.75/0.25.
        # Feel free to change it to other parameters if that gives you better performance
        for i in range(self.state_size-1):
            self.trans_logpdf[i][i] = np.log(0.75)
            self.trans_logpdf[i][i+1] = np.log(0.25)

        # initialize emission model
        for i in range(self.state_size):
            frame_lst = self.state2frames[i]
            frames = np.stack(frame_lst)
            self.gmms[i].initialize(frames)

        self.clean_accumulation()

    def clean_accumulation(self):
        """
        After one EM step, you might want to clean your existing alignments.

        Returns:

        """
        self.state2frames = [[] for i in range(self.state_size)]
        self.state2count = [[0, 0] for i in range(self.state_size)]

    def align_equally(self, X: np.ndarray):
        """
        Before training the GMM-HMM model, we would like to obtain a good initialization parameters especially for the GMM.
        We assume that each state is all equally aligned with the observation.
        For example if the number of frames in X is 10 and there are 5 states,
        we might want to align X with [0,0,1,1,2,2,3,3,4,4]

        Args:
            X (np.ndarray): a single sample point of [N,D] where $N$ is the number of frames and $D$ is the feature size

        Returns:
            alignments (np.ndarray): one dimension numpy array with shape [N], each element should be an int between [0, state_size-1]

        """

        assert len(X.shape) == 2, "sample should have two dimension"
        assert X.shape[1] == self.feature_size, "feature size does not match"

        frame_size = X.shape[0]
        frame_per_state = int(frame_size / self.state_size)

        alignment = np.ones(frame_size, dtype=np.int32)*(self.state_size-1)

        for i in range(self.state_size):
            alignment[i*frame_per_state:(i+1)*frame_per_state] = i

        return alignment

    def align(self, X: np.ndarray) -> np.ndarray:
        """
        This is the E step of HMM, in which we would like to align each frame in X to a HMM state.
        Implement the Viterbi search described in the handout to find the good alignment.
        You might want to do most of the computation in the log space by using logpdf in GMM and trans_logprob

        Args:
            X (np.ndarray): a single sample point of [N,D] where $N$ is the number of frames and $D$ is the feature size

        Returns:
            alignments (np.ndarray): one dimension numpy array with shape [N], each element should be an int between [0, state_size-1]
            You can check the returned value of align_equally as an example.
        """

        time_size = X.shape[0]
        feature_size = X.shape[1]

        assert self.feature_size == feature_size

        # compute all probs at once
        emission_pdf = np.zeros((self.state_size, time_size))
        for i, gmm in enumerate(self.gmms):
            emission_pdf[i] = gmm.logpdf(X)

        # viterbi search graph
        graph = -np.inf*np.ones((self.state_size, time_size))

        # rembmer max previous pointer
        prev_pointer = np.zeros((self.state_size, time_size), dtype=np.int32)

        # start point is (0,0)
        graph[0][0] = 0.0

        # implement your viterbi search
        for s in range(self.state_size):
            graph[s, 0] = self.trans_logpdf[s, s] + emission_pdf[s, 0]
        for t in range(1, time_size):
            for s in range(self.state_size):
                max_prob = np.max(graph[:, t-1] + self.trans_logpdf[:, s])
                graph[s, t] = max_prob + emission_pdf[s, t]
                prev_pointer[s, t] = np.argmax(graph[:, t-1] + self.trans_logpdf[:, s])
        s_T = np.argmax(graph[:, time_size-1])
        viterbi_align = np.zeros(time_size, dtype=np.int32)
        viterbi_align[-1] = s_T
        for t in range((time_size - 1), 0, -1):
            s_t_1 = viterbi_align[t]
            s_t = prev_pointer[s_t_1, t]
            viterbi_align[t - 1] = s_t
            
        return viterbi_align
    
    def accumulate(self, X: np.ndarray, alignments: np.ndarray) -> None:
        """
        Accumulate the aligned frames/counts for each state. Those statistics would be latter used to update GMM/HMM

        Args:
            X (np.ndarray): sample matrix, it is a [N, H] shape matrix where N is the frame size, H is the feature size
            alignments (np.ndarray): alignment returned from align or align_equally. it is a [N] shaped numpy array.

        Returns:
        """

        prev_id = -1

        for i, state_id in enumerate(alignments):

            # accumulate stats for emission
            self.state2frames[state_id].append(X[i])

            # accumulate stats for transition
            if prev_id >= 0:
                if prev_id == state_id:

                    # inc transition from prev_id to itself
                    self.state2count[prev_id][0] += 1
                else:
                    # inc transition from prev_id to next id
                    self.state2count[prev_id][1] += 1

            prev_id = state_id

    def update(self):
        """
        using existing accumulated alignments to update GMM and HMM.
        Do not forget to clean your accumulation

        Returns:
        """

        # update GMMs
        for s in range(self.state_size):
            # collect observations o_t generated from state s
            X_i = np.vstack(self.state2frames[s])
            self.gmms[s].fit(X_i)
    
        for s in range(self.state_size):
            if s != self.state_size - 1:
                if self.state2count[s][0] > 0:
                    self.trans_logpdf[s, s] = np.log(self.state2count[s][0] / (self.state2count[s][0] + self.state2count[s][1]))
                if self.state2count[s][1] > 0:
                    self.trans_logpdf[s, s + 1] = np.log(self.state2count[s][1] / (self.state2count[s][0] + self.state2count[s][1]))
            else:
                if self.state2count[s][0] > 0:
                    self.trans_logpdf[s, s] = np.log(self.state2count[s][0] / (self.state2count[s][0] + self.state2count[s][1]))
        self.clean_accumulation()

    def logpdf(self, X):
        """
        compute the marginalized log probability of X

        Args:
            X (): sample matrix, it is a [N, H] shape matrix where N is the frame size, H is the feature size

        Returns: log probability
        """

        time_size = X.shape[0]
        feature_size = X.shape[1]

        assert self.feature_size == feature_size

        # compute all probs at once
        emission_pdf = np.zeros((self.state_size, time_size))
        for i, gmm in enumerate(self.gmms):
            emission_pdf[i] = gmm.logpdf(X)

        # viterbi search graph
        graph = -np.inf*np.ones((self.state_size, time_size))

        # start point is (0,0)
        graph[0][0] = 0.0

        # forward path (almost similar to the viterbi search)
        for s in range(self.state_size):
            graph[s, 0] = self.trans_logpdf[s, s] + emission_pdf[s, 0]
        for t in range(1, time_size):
            for s in range(self.state_size):
                graph[s, t] = emission_pdf[s, t] + logsumexp(self.trans_logpdf[:, s] + graph[:, t-1])
        
        log_prob = logsumexp(graph[:, -1])

        return log_prob
