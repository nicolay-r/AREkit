from arekit.contrib.source.rusentrel.synonyms_helper import RuSentRelSynonymsCollectionHelper
from arekit.processing.lemmatization.mystem import MystemWrapper


def read_rusentrel_synonyms_collection(version):
    # Initializing stemmer
    stemmer = MystemWrapper()

    # Reading synonyms collection.
    return RuSentRelSynonymsCollectionHelper.load_collection(stemmer=stemmer,
                                                             version=version)

