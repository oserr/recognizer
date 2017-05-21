import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    len_all = len(test_set.get_all_Xlengths())
    for word_id in range(0, len_all):
        x, lens = test_set.get_item_Xlengths(word_id)
        probs = {}
        for word, model in models.items():
            try:
                logl = model.score(x, lens)
                probs[word] = logl
            except:
                probs[word] = float('-inf')
        probabilities.append(probs)
        w, _ = max(probs.items(), key=lambda x: x[1])
        guesses.append(w)
    return (probabilities, guesses)
