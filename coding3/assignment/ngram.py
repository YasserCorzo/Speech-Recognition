from collections import defaultdict, Counter
from typing import List

class KneserNeyLM:

    def __init__(self, d=0.75):
        """
        KneserNey Language Modeling

        Args:
            d (int): the discounting parameter. We will fix it to 0.75 in this assignment
        """

        # discount parameter
        self.d = d

    def fit(self, sentences: List[str]) -> None:
        """
        Estimate the language model from training sentences

        Args:
            sentences (List[str]): a list of sentence. Each sentence is a str.
            <s> and </s> are not included in the sentences, make sure you add both <s> and </s> to each sentence

        Returns:
        """
        # add <s> (start of sentence) and </s> (end of sentence) tokens
        # TODO:
        self.sentences = []
        n_gram_l = []
        
        self.unigram_vocab = set()
        self.bigram_vocab = set()
        self.trigram_vocab = set()
       
        self.unigram_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        self.trigram_count = defaultdict(int)

        for sentence in sentences:
            sentence = '<s> ' + sentence + ' </s>'
            self.sentences.append(sentence)

        # calculate C for unigrams
        for sentence in self.sentences:
            for s in sentence.split(" "):
                self.unigram_count[s] += 1
            
            # add letters in sentence to unigram vocab
            self.unigram_vocab |= set(sentence.split(" "))

        # calculate C for bigram and bigram vocab (will be used for N1+)
        for sentence in self.sentences:
            s = sentence.split(" ")
            for i in range((len(s)) - 1):
                ith_n_gram = " ".join(s[i : i + 2])
                self.bigram_count[ith_n_gram] += 1
                n_gram_l.append(ith_n_gram)
        
        self.bigram_vocab |= set(n_gram_l)
        n_gram_l = []

        # calculate C for trigram and trigram vocab (will be used for N1+)
        for sentence in self.sentences:
            s = sentence.split(" ")
            for i in range((len(s)) - 2):
                ith_n_gram = " ".join(s[i : i + 3])
                self.trigram_count[ith_n_gram] += 1
                n_gram_l.append(ith_n_gram)
        
        self.trigram_vocab |= set(n_gram_l)

    def unigram_prob(self, unigram: List[str]) -> float:
        """
        Compute the simple count based probability P(w_1),
        return a small number if w_1 is not appearing in the dataset

        Args:
            unigram ([List[str])): one word

        Returns:
        """
        # TODO:

        if self.unigram_count[unigram[0]] == 0:
            # 1 / vocab size
            return 1 / len(self.unigram_vocab)
        
        total_count = 0
        for w in self.unigram_count:
            total_count += self.unigram_count[w]
        
        return self.unigram_count[unigram[0]] / total_count

    def bigram_prob(self, bigram: List[str]) -> float:
        """
        Compute the KneserNey bigram probability P(w_2|w_1)

        Args:
            bigram (List[str]): two words: [w_1, w_2]

        Returns: probability
        """
        # TODO:
        w_1, w_2 = bigram
        N1_w1_star = 0
        for bigram in self.bigram_vocab:
            if bigram.split(" ")[0] == w_1:
                N1_w1_star += 1
        
        lamba_w1 = (self.d * N1_w1_star) / self.unigram_count[w_1]   

        N1_star_w2 = 0
        num_unique_bigrams = len(self.bigram_vocab)
        for bigram in self.bigram_vocab:
            if bigram.split(" ")[1] == w_2:
                N1_star_w2 += 1
        
        p_cont_w_2 = N1_star_w2 / num_unique_bigrams

        bigram = w_1 + " " + w_2
        bigram_prob = (max(self.bigram_count[bigram] - self.d, 0) / self.unigram_count[w_1]) + (lamba_w1 * p_cont_w_2)
        
        return bigram_prob

    def trigram_prob(self, trigram: List[str]) -> float:
        """
        Compute the KneserNey trigram probability P(w_3|w_1, w_2)

        Args:
            trigram (List[str]): three words: [w_1, w_2, w_3]

        Returns: probability
        """
        # TODO:
        w_1, w_2, w_3 = trigram
        N1_w1_w2_star = 0
        for trigram in self.trigram_vocab:
            if trigram.split(" ")[:-1] == [w_1, w_2]:
                N1_w1_w2_star += 1
        
        lamba_w1_w2 = (self.d * N1_w1_w2_star) / self.bigram_count[w_1 + " " + w_2]   

        N1_star_w3 = 0
        num_unique_bigrams = len(self.bigram_vocab)
        for bigram in self.bigram_vocab:
            if bigram.split(" ")[1] == w_3:
                N1_star_w3 += 1
        
        p_cont_w_3 = N1_star_w3 / num_unique_bigrams

        N1_w2_star = 0
        for bigram in self.bigram_vocab:
            if bigram.split(" ")[0] == w_2:
                N1_w2_star += 1

        N1_star_w2_star = 0
        for trigram in self.trigram_vocab:
            if trigram.split(" ")[1] == w_2:
                N1_star_w2_star += 1
        
        lambda_w2 = (self.d * N1_w2_star) / N1_star_w2_star

        N1_star_w2_w3 = 0
        for trigram in self.trigram_vocab:
            if trigram.split(" ")[1:] == [w_2, w_3]:
                N1_star_w2_w3 += 1
        
        p_cont_w3_w2 = (max(N1_star_w2_w3 - self.d, 0) / N1_star_w2_star) + (lambda_w2 * p_cont_w_3)

        bigram = w_1 + " " + w_2
        trigram = w_1 + " " + w_2 + " " + w_3
        trigram_prob = (max(self.trigram_count[trigram] - self.d, 0) / self.bigram_count[bigram]) + (lamba_w1_w2 * p_cont_w3_w2)
        
        return trigram_prob


    def sentence_prob(self, sentence: str) -> float:
        """
        compute perplexity for each sentence

        Args:
            sentence (str): a sentence. <s> and </s> are not included in the sentence

        Returns:
            probability (float)
        """
        # add <s> (start of sentence) and </s> (end of sentence) tokens

        # TODO:
        sentence = '<s> ' + sentence + ' </s>'
        sentence_prob = 1
        s = sentence.split(" ")
        for i in range((len(s)) - 2):
            if i == 0:
                sentence_prob *= self.bigram_prob(s[i : i + 2])
            sentence_prob *= self.trigram_prob(s[i : i + 3])

        return sentence_prob
    
if __name__ == "__main__":
    # python hw3/ngram.py
    lm = KneserNeyLM()
    lm.fit(['a b c a b'])

    assert lm.unigram_prob(['a']) == 0.2857142857142857

    assert lm.bigram_prob(['a', 'b']) == 0.7

    assert lm.trigram_prob(['a', 'b', 'c']) == 0.33125000000000004

    assert lm.sentence_prob('a b c') == 0.015884472656250002

    print("Passed 4/4 tests!")