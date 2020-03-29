from arekit.contrib.experiments.cv.doc_stat.base import DocStatGeneratorBase


class RuAttitudesStatGenerator(DocStatGeneratorBase):

    def __init__(self, synonyms):
        self.__synonyms = synonyms

    def iter_doc_ids(self):
        raise NotImplementedError()

    def calculate_sentences_count(self, doc_id):
        raise NotImplementedError()
