from os import path

from arekit.common.entities.base import Entity
from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.news import News
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.processing.text.token import Token

NewsTermsShow = False
NewsTermsStatisticShow = False


class BaseExperiment(object):

    def __init__(self, data_io, opin_operation, doc_operations, prepare_model_root):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(prepare_model_root, bool))
        assert(isinstance(opin_operation, OpinionOperations))
        assert(isinstance(doc_operations, DocumentOperations))

        self.__opin_operations = opin_operation
        self.__doc_operations = doc_operations

        self.__data_io = data_io

        if prepare_model_root:
            self.DataIO.prepare_model_root()

        self.__neutral_annot = self.__init_annotator()

        # Setup DataIO
        # TODO. Move into data_io
        self.__data_io.Callback.set_log_dir(log_dir=path.join(self.DataIO.get_model_root(), u"log/"))

        self.__neutral_annot.initialize(data_io=data_io,
                                        opin_ops=self.OpinionOperations,
                                        doc_ops=self.DocumentOperations)

        self.__data_io.ModelIO.set_model_root(value=self.DataIO.get_model_root())

    # region Properties

    @property
    def DataIO(self):
        return self.__data_io

    @property
    def NeutralAnnotator(self):
        return self.__neutral_annot

    @property
    def OpinionOperations(self):
        return self.__opin_operations

    @property
    def DocumentOperations(self):
        return self.__doc_operations

    # endregion

    def create_parsed_collection(self, data_type):
        assert(isinstance(data_type, unicode))

        parsed_collection = ParsedNewsCollection()

        for doc_id in self.DocumentOperations.iter_news_indices(data_type):

            news = self.DocumentOperations.read_news(doc_id=doc_id)
            assert(isinstance(news, News))

            parsed_news = news.parse(options=self.DocumentOperations.create_parse_options())

            if NewsTermsStatisticShow:
                self.__debug_statistics(parsed_news)
            if NewsTermsShow:
                self.__debug_show_terms(parsed_news)

            # TODO. Remove this (move into other place)
            parsed_news.modify_parsed_sentences(
                lambda sentence: FrameVariantsParser.parse_frames_in_parsed_text(
                    frame_variants_collection=self.DataIO.FrameVariantCollection,
                    parsed_text=sentence))

            if not parsed_collection.contains_id(doc_id):
                parsed_collection.add(parsed_news)
            else:
                print "Warning: Skipping document with id={}, news={}".format(news.ID, news)

        return parsed_collection

    # region private methods

    # TODO. Move to post processing stat.
    @staticmethod
    def __debug_show_terms(parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        for term in parsed_news.iter_terms():
            if isinstance(term, unicode):
                print "Word:    '{}'".format(term.encode('utf-8'))
            elif isinstance(term, Token):
                print "Token:   '{}' ('{}')".format(term.get_token_value().encode('utf-8'),
                                                    term.get_original_value().encode('utf-8'))
            elif isinstance(term, Entity):
                print "Entity:  '{}'".format(term.Value.encode('utf-8'))
            else:
                raise Exception("unsuported type {}".format(term))

    # TODO. Move to post processing stat.
    @staticmethod
    def __debug_statistics(parsed_news):
        assert(isinstance(parsed_news, ParsedNews))

        terms = list(parsed_news.iter_terms())
        words = filter(lambda term: isinstance(term, unicode), terms)
        tokens = filter(lambda term: isinstance(term, Token), terms)
        entities = filter(lambda term: isinstance(term, Entity), terms)
        total = len(words) + len(tokens) + len(entities)

        print "Extracted news_words info, NEWS_ID: {}".format(parsed_news.RelatedNewsID)
        print "\tWords: {} ({}%)".format(len(words), 100.0 * len(words) / total)
        print "\tTokens: {} ({}%)".format(len(tokens), 100.0 * len(tokens) / total)
        print "\tEntities: {} ({}%)".format(len(entities), 100.0 * len(entities) / total)

    def __init_annotator(self):
        if isinstance(self.__data_io.LabelsScaler, TwoLabelScaler):
            return TwoScaleNeutralAnnotator()
        if isinstance(self.__data_io.LabelsScaler, ThreeLabelScaler):
            return ThreeScaleNeutralAnnotator()

    # endregion
