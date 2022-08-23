from arekit.common.text.stemmer import Stemmer
from arekit.contrib.utils.download import NEWS_MYSTEM_SKIPGRAM_1000_20_2015, load_embedding_and_vocab
from arekit.contrib.utils.embeddings.rusvectores import RusvectoresEmbedding


def load_embedding_news_mystem_skipgram_1000_20_2015(stemmer):
    """ Embedding from https://rusvectores.org/ru/models/
        Description:        Russian news, from 2013 till the october 2015
        Corpora size:       2.5 milliard words
        Vocabulary volume:  147 358
        Frequency bound:    200
        Tagset:             Mystem
        Algorithm:          Continuous Skip-Gram
        Vector size:        1000

        stemmer: Stemmer
            It is expected to adopt MystemWrapper.
    """
    assert(isinstance(stemmer, Stemmer) or stemmer is None)
    embedding, vocab = load_embedding_and_vocab(local_name=NEWS_MYSTEM_SKIPGRAM_1000_20_2015, check_existance=True)
    embedding = RusvectoresEmbedding(matrix=embedding, words=vocab, stemmer=stemmer)
    return embedding
