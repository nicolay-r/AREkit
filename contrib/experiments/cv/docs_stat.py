from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news import RuSentRelNews
from tests.test_cv import synonyms, stat_filepath


def DocumentStatGeneratorBase(object):
    """
    Provides statistic on certain document.
    Abstract, considered a specific implementation for document processing operation.
    """

    # TODO. Complete with other methods

    def get_sentences_count(doc_id):
        raise NotImplementedError()


# TODO. Move in Generator as a method
def read_docs_stat(stat_filepath):
    """
    return:
        list of the following pairs: (doc_id, sentences_count)
    """
    docs_info = []
    with open(stat_filepath, 'r') as f:
        for line in f.readlines():
            args = [int(i) for i in line.split(':')]
            doc_id, s_count = args
            docs_info.append((doc_id, s_count))

    return docs_info


# TODO. Use it in _io.py
# TODO. Move in Generator as a method
def write_doc_stat():

    with open(stat_filepath, 'w') as f:
        for doc_index, s_count in __iter_rusentrel_stat():
            f.write("{}: {}\n".format(doc_index, s_count))


def __iter_rusentrel_stat():

    # TODO. Now it is RuSentRel statistics.
    # TODO. Refactoring.
    # TODO. Implement in doc_stat_generator.
    # TODO. Make not iterative

    for doc_id in RuSentRelIOUtils.iter_collection_indices():

        entities = RuSentRelDocumentEntityCollection.read_collection(
            doc_id=doc_id,
            synonyms=synonyms)

        news = RuSentRelNews.read_document(doc_id=doc_id,
                                           entities=entities)

        yield (doc_id, news.SentencesCount())
