import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        skip_connect = [0]
        for i, symbol in enumerate(target):
            if i > 0 and target[i] != target[i-1]:
                skip_connect.append(1)
            else:
                skip_connect.append(0)
            skip_connect.append(0)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        # empty initialization of alpha
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))
        
        # TODO:
        
        # initializing starting alpha of trellis path
        alpha[0][0] = logits[0][0]
        alpha[0][1] = logits[0][1]
        
        # initialize alphas at first extended symbol (u_i = 0)
        for t in range(1, T):
            alpha[t][0] = alpha[t - 1][0] * logits[t][extended_symbols[0]]
            
        for t in range(1, T):
            for u_i in range(1, S):
                if skip_connect[u_i]:
                    alpha[t][u_i] = (alpha[t - 1][u_i] + alpha[t - 1][u_i - 1] + alpha[t - 1][u_i - 2]) * logits[t][extended_symbols[u_i]]
                else:
                    alpha[t][u_i] = (alpha[t - 1][u_i] + alpha[t - 1][u_i - 1]) * logits[t][extended_symbols[u_i]]
        
        return alpha
                


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """
        # empty initialization of beta
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # TODO:
        
        # initializing beta of end of trellis path
        beta[-1][-1] = 1
        beta[-1][-2] = 1
        
        # initialize betas at last extended symbol (u_i = S - 1)
        for t in range(T - 2, -1, -1):
            beta[t][-1] = beta[t + 1][-1] * logits[t + 1][extended_symbols[-1]]
           
        for t in range(T - 2, -1, -1):
            for u_i in range(S - 2, -1, -1):
                if u_i + 2 < S and skip_connect[u_i + 2]:
                    beta[t][u_i] = beta[t + 1][u_i] * logits[t + 1][extended_symbols[u_i]] + beta[t + 1][u_i + 1] * logits[t + 1][extended_symbols[u_i + 1]] + beta[t + 1][u_i + 2] * logits[t + 1][extended_symbols[u_i + 2]]
                else:
                    beta[t][u_i] = beta[t + 1][u_i] * logits[t + 1][extended_symbols[u_i]] + beta[t + 1][u_i + 1] * logits[t + 1][extended_symbols[u_i + 1]]
        
        return beta
        

    def get_occupation_probability(self, alpha, beta):
        """Compute occupation probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        # empty initialization of gamma
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))

        # TODO:
        gamma = alpha * beta
        return gamma

if __name__ == "__main__":
    # python hw3/ctc.py
    # local tests for alpha, beta, and occupation probability
    ctc = CTC()
    target = [1,2,2,3]  # A B B C
    extSymbols, skipConnect = ctc.extend_target_with_blank(target)
    logits = np.array([[.7, .2, .05, .05],  #<b>
                       [.7, .2, .05, .05],  #<b>
                       [.1, .8, .05, .05],  #A
                       [.1, .8, .05, .05],  #A
                       [.8, .05, .1, .05],  #<b>
                       [.8, .05, .1, .05],  #<b>
                       [.1, .05, .8, .05],  #B
                       [.1, .05, .8, .05],  #B
                       [.8, .05, .1, .05],  #<b>
                       [.1, .05, .8, .05],  #B
                       [.1, .05, .05, .8],  #C
                       [.1, .05, .05, .8],  #C
                       ])
    
    alpha = ctc.get_forward_probs(logits, extSymbols, skipConnect)
    expected_alpha = np.array([[7.00000000e-01, 2.00000000e-01, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00],
                            [4.90000000e-01, 1.80000000e-01, 1.40000000e-01, 1.00000000e-02,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00],
                            [4.90000000e-02, 5.36000000e-01, 3.20000000e-02, 1.65000000e-02,
                                1.00000000e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00],
                            [4.90000000e-03, 4.68000000e-01, 5.68000000e-02, 2.92250000e-02,
                                1.75000000e-03, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00],
                            [3.92000000e-03, 2.36450000e-02, 4.19840000e-01, 5.54025000e-02,
                                2.47800000e-02, 1.80000000e-04, 4.00000000e-05, 2.50000000e-06,
                                0.00000000e+00],
                            [3.13600000e-03, 1.37825000e-03, 3.54788000e-01, 4.98887500e-02,
                                6.41460000e-02, 2.49600000e-03, 1.76000000e-04, 1.11250000e-05,
                                2.00000000e-06],
                            [3.13600000e-04, 2.25712500e-04, 3.56166250e-02, 3.24844000e-01,
                                1.14034750e-02, 5.33136000e-02, 2.67200000e-04, 1.34156250e-04,
                                1.31250000e-06],
                            [3.13600000e-05, 2.69656250e-05, 3.58423375e-03, 2.88549070e-01,
                                3.36247475e-02, 5.17736600e-02, 5.35808000e-03, 2.68574781e-03,
                                1.35468750e-05],
                            [2.50880000e-05, 2.91628125e-06, 2.88895950e-03, 2.92160269e-02,
                                2.57739054e-01, 8.53984075e-03, 4.57053920e-02, 2.99087439e-03,
                                2.15943575e-03],
                            [2.50880000e-06, 1.40021406e-06, 2.89187578e-04, 2.56863222e-02,
                                2.86955081e-02, 2.13023116e-01, 5.42452328e-03, 2.86180536e-03,
                                5.15031014e-04],
                            [2.50880000e-07, 1.95450703e-07, 2.90587792e-05, 1.29884550e-03,
                                5.43818303e-03, 1.20859312e-02, 2.18447639e-02, 1.77047556e-01,
                                3.37683637e-04],
                            [2.50880000e-08, 2.23165352e-08, 2.92542299e-06, 6.64049864e-05,
                                6.73702853e-04, 8.76205711e-04, 3.39306951e-03, 1.68782601e-01,
                                1.77385239e-02]])
    assert np.allclose(alpha, expected_alpha), "Did not get expected alpha value"

    beta = ctc.get_backward_probs(logits, extSymbols, skipConnect)
    expected_beta = np.array([[2.50651609e-01, 5.53249903e-02, 3.34539600e-03, 8.01323284e-04,
                                7.12372327e-04, 5.13686352e-06, 4.42621064e-06, 4.03658398e-08,
                                3.58400000e-08],
                            [2.83817164e-01, 2.59897972e-01, 4.65104913e-03, 1.79323220e-03,
                                1.01665953e-03, 1.42130574e-05, 6.31669258e-06, 9.05167969e-08,
                                5.12000000e-08],
                            [2.85702416e-01, 3.19058653e-01, 3.86658010e-02, 1.56893807e-02,
                                1.00876317e-02, 1.57927297e-04, 6.27737578e-05, 7.86335938e-07,
                                5.12000000e-07],
                            [5.30956400e-02, 3.50491065e-01, 3.29688985e-01, 1.13938050e-01,
                                9.99247816e-02, 1.90307078e-03, 6.24994219e-04, 5.48671875e-06,
                                5.12000000e-06],
                            [4.03669500e-02, 4.16041600e-01, 3.92997050e-01, 1.52913450e-01,
                                1.23308381e-01, 1.27807656e-02, 7.80784375e-04, 7.33437500e-06,
                                6.40000000e-06],
                            [2.16530000e-02, 4.60891000e-01, 4.39240000e-01, 4.16050500e-01,
                                1.39135500e-01, 1.19999812e-01, 9.74812500e-04, 1.86875000e-05,
                                8.00000000e-06],
                            [2.00000000e-05, 4.33020000e-01, 4.33000000e-01, 4.94925000e-01,
                                2.01105000e-01, 1.48781250e-01, 9.64125000e-03, 2.13750000e-04,
                                8.00000000e-05],
                            [0.00000000e+00, 4.00000000e-04, 4.00000000e-04, 5.41200000e-01,
                                6.19650000e-01, 1.73925000e-01, 9.50750000e-02, 2.67500000e-03,
                                8.00000000e-04],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.00000000e-03,
                                6.76000000e-01, 7.88500000e-01, 1.16500000e-01, 3.75000000e-02,
                                1.00000000e-03],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                4.00000000e-02, 8.40000000e-01, 8.00000000e-01, 7.30000000e-01,
                                1.00000000e-02],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, 8.00000000e-01, 8.00000000e-01, 9.00000000e-01,
                                1.00000000e-01],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
                                1.00000000e+00]])
    assert np.allclose(beta, expected_beta), "Did not get expected beta value"

    occupation_prob = ctc.get_occupation_probability(alpha, beta)
    expected_occupation_prob = np.array([[1.75456126e-01, 1.10649981e-02, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00],
                                        [1.39070410e-01, 4.67816349e-02, 6.51146879e-04, 1.79323220e-05,
                                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00],
                                        [1.39994184e-02, 1.71015438e-01, 1.23730563e-03, 2.58874781e-04,
                                            1.00876317e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00],
                                        [2.60168636e-04, 1.64029818e-01, 1.87263343e-02, 3.32983951e-03,
                                            1.74868368e-04, 9.51535391e-08, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00],
                                        [1.58238444e-04, 9.83730363e-03, 1.64995881e-01, 8.47178741e-03,
                                            3.05558169e-03, 2.30053781e-06, 3.12313750e-08, 1.83359375e-11,
                                            0.00000000e+00],
                                        [6.79038080e-05, 6.35223021e-04, 1.55837081e-01, 2.07562394e-02,
                                            8.92498578e-03, 2.99519532e-04, 1.71567000e-07, 2.07898438e-10,
                                            1.60000000e-11],
                                        [6.27200000e-09, 9.77380268e-05, 1.54219986e-02, 1.60773417e-01,
                                            2.29329584e-03, 7.93206405e-03, 2.57614200e-06, 2.86758984e-08,
                                            1.05000000e-10],
                                        [0.00000000e+00, 1.07862500e-08, 1.43369350e-06, 1.56162757e-01,
                                            2.08355748e-02, 9.00473382e-03, 5.09419456e-04, 7.18437540e-06,
                                            1.08375000e-08],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.16864108e-04,
                                            1.74231601e-01, 6.73366443e-03, 5.32467817e-03, 1.12157790e-04,
                                            2.15943575e-06],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            1.14782032e-03, 1.78939417e-01, 4.33961862e-03, 2.08911791e-03,
                                            5.15031014e-06],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00, 9.66874496e-03, 1.74758111e-02, 1.59342800e-01,
                                            3.37683637e-05],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.68782601e-01,
                                            1.77385239e-02]])
    assert np.allclose(occupation_prob, expected_occupation_prob), "Did not get expected occupation probability value"
    
    print("Passed 3/3 tests!")