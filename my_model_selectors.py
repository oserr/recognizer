import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def compute_for_all_n(self):
        '''Computes the model and log likelihood for all combinations of
        components.

        :return
            A list of tuples of the form (n, logL), where n is the number
            of components used to train the model, and logL is the log
            likelihood for the given model.
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cross_n_results = []
        for n in range(self.min_n_components, self.max_n_components+1):
            len_sequence = len(self.sequences)
            split_method = KFold(len_sequence if len_sequence<3  else 3)
            list_logl = []  # list of cross validation scores obtained

            for train_idx, test_idx in split_method.split(self.sequences):
                x, lens = combine_sequences(train_idx, self.sequences)
                try:
                    model = self.compute_model(n, x, lens)
                except:
                    continue
                x, lens = combine_sequences(test_idx, self.sequences)
                try:
                    logl = model.score(x, lens)
                except:
                    continue
                list_logl.append(logl)

            if list_logl:
                avg_logl = np.mean(list_logl)
                cross_n_results.append((n, avg_logl))
        return cross_n_results

    def compute_model(self, n, x, lens):
        '''Computes a GaussianHMM model.

        :param n
            The number of components.
        :param x
            The observations.
        :param lens
            The lengths of each sequence of observations.
        '''
        return GaussianHMM(
            n_components=n,
            covariance_type="diag",
            n_iter=1000,
            random_state=self.random_state,
            verbose=False
        ).fit(x, lens)


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    '''Select best model based on average log Likelihood of cross-validation
    folds.
    '''

    def select(self):
        '''Selects cross-validated model with best log likelihood result.'''
        results = self.compute_for_all_n()
        n, _ = max(results, key=lambda x: x[1])
        return self.compute_model(n, self.X, self.lengths)
