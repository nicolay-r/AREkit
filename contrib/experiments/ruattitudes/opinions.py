from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.contrib.source.ruattitudes.news.helper import RuAttitudesNewsHelper


class RuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, synonyms, ru_attitudes):
        assert(isinstance(ru_attitudes, dict) or ru_attitudes is None)
        super(RuAttitudesOpinionOperations, self).__init__(synonyms)

        self.__ru_attitudes = ru_attitudes

        # We consider that the neutral annotation to be performed in advance
        # in RuAttitudes collection. Version 1.0 does not provide the latter
        # but it might be provided manually or since version 2.0.
        self.__neutrally_annot_doc_ids = set()

    # region private methods

    def __get_opinions_in_news(self, doc_id, opinion_check=lambda _: True):
        news = self.__ru_attitudes[doc_id]
        return [opinion
                for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)
                if opinion_check(opinion)]

    # endregion

    def iter_opinions_for_extraction(self, doc_id, data_type):
        if data_type == DataType.Train:
            # We provide only those opinions that
            # were labeled in RuAttitudes collection
            yield self.__get_opinions_in_news(doc_id=doc_id)

        # Provide nothing otherwise
        pass

    def get_doc_ids_set_to_neutrally_annotate(self):
        return self.__neutrally_annot_doc_ids

    def read_etalon_opinion_collection(self, doc_id):
        return self.__get_opinions_in_news(doc_id=doc_id)
