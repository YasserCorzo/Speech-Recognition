import numpy as np
from collections import defaultdict

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols

        """
        # blank is index 0
        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(seq_length, len(symbols) + 1)]

        Returns
        -------

        output [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols
        """

        # TODO:
        argmax_zt = np.argmax(y_probs, axis=1)
        select_i = np.ones(len(argmax_zt), dtype=bool)

        # calculating positions where following value is repeated
        select_i[1:] = argmax_zt[1:] != argmax_zt[:-1]

        # calculate positions of non-zero values
        select_i &= argmax_zt != 0 

        # retrieve symbol sequence indices
        compressed_symbols_i = argmax_zt[select_i]

        # retrieve compressed symbol sequence
        compressed_symbols_seq = ""
        for i in compressed_symbols_i:
            compressed_symbols_seq += self.symbol_set[i]
        
        return compressed_symbols_seq


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_size, lm_weight, ngram):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """
        # blank is index 0
        self.symbol_set = symbol_set
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        # ngram.fit(<text_data>) has already been called by decode.py. this function only needs ngram.sentence_prob()
        self.ngram = ngram

    def decode(self, y_probs):
        # TODO
        raise NotImplementedError
    

    if __name__ == "__main__":
        greedy_search = GreedySearchDecoder(['<b>','A','B','C'])

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
        
        output = greedy_search.decode(logits)
        assert output == "ABBC"
        print("Passed 1/1 tests!")