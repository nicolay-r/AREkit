from arekit.common.experiment.cv.doc_stat.base import DocStatGeneratorBase


class RuAttitudesStatGenerator(DocStatGeneratorBase):

    def __init__(self, synonyms):
        self.__synonyms = synonyms

    def _iter_doc_ids(self):
        raise NotImplementedError()

    def _calculate_sentences_count(self, doc_id):
        raise NotImplementedError()
