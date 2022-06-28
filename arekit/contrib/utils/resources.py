from arekit.contrib.utils.download import get_resource_path, NEWS_MYSTEM_SKIPGRAM_1000_20_2015
from arekit.contrib.utils.embeddings.rusvectores import RusvectoresEmbedding


def load_embedding_news_mystem_skipgram_1000_20_2015():
    """ Embedding from https://rusvectores.org/ru/models/
        Description:        Russian news, from 2013 till the october 2015
        Corpora size:       2.5 milliard words
        Vocabulary volume:  147 358
        Frequency bound:    200
        Tagset:             Mystem
        Algorithm:          Continuous Skip-Gram
        Vector size:        1000
    """
    filepath = get_resource_path(local_name=NEWS_MYSTEM_SKIPGRAM_1000_20_2015,
                                 check_existance=True)
    return RusvectoresEmbedding.from_word2vec_format(filepath=filepath, binary=True)
