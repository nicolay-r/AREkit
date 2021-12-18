from collections import OrderedDict

from arekit.common.data import const
from arekit.common.data.input.providers.instances.multiple import MultipleLinkedTextOpinionsInstancesProvider
from arekit.common.data.input.providers.instances.single import SingleInstanceTextOpinionsLinkageProvider
from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.labels.base import Label

from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.bert.input.providers.label_binary import BinaryLabelProvider
from arekit.contrib.bert.input.providers.row_ids_binary import BinaryIDProvider


class BaseSampleRowProvider(BaseRowProvider):
    """ Rows provider for samples storage.
    """

    def __init__(self, label_provider, text_provider):
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(text_provider, BaseSingleTextProvider))
        super(BaseSampleRowProvider, self).__init__()

        self._label_provider = label_provider
        self.__text_provider = text_provider
        self.__row_ids_provider = self.__create_row_ids_provider(label_provider)
        self.__instances_provider = self.__create_instances_provider(label_provider)
        self.__store_labels = None

    # region properties

    @property
    def LabelProvider(self):
        return self._label_provider

    @property
    def TextProvider(self):
        return self.__text_provider

    # endregion

    # region protected methods

    @staticmethod
    def _provide_sentence_terms(parsed_news, sentence_ind):
        return parsed_news.iter_sentence_terms(sentence_index=sentence_ind, return_id=False)

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(self.__store_labels, bool))

        def __assign_value(column, value):
            row[column] = value

        row[const.ID] = self.__row_ids_provider.create_sample_id(
            linked_opinions=text_opinion_linkage,
            index_in_linked=index_in_linked,
            label_scaler=self._label_provider.LabelScaler)

        row[const.DOC_ID] = text_opinion_linkage.First.DocID

        expected_label = text_opinion_linkage.get_linked_label()

        if self.__store_labels:
            row[const.LABEL] = self._label_provider.calculate_output_uint_label(
                expected_uint_label=self._label_provider.LabelScaler.label_to_uint(expected_label),
                etalon_uint_label=self._label_provider.LabelScaler.label_to_uint(etalon_label))

        self.__text_provider.add_text_in_row(
            set_text_func=lambda column, value: __assign_value(column, value),
            sentence_terms=list(self._provide_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind)),
            s_ind=s_ind,
            t_ind=t_ind,
            expected_label=expected_label)

        row[const.S_IND] = s_ind
        row[const.T_IND] = t_ind

    def _provide_rows(self, parsed_news, text_opinion_linkage, idle_mode):
        assert(isinstance(idle_mode, bool))

        row_dict = OrderedDict()

        for index_in_linked in range(len(text_opinion_linkage)):

            rows_it = self.__provide_rows(
                parsed_news=parsed_news,
                row_dict=row_dict,
                text_opinion_linkage=text_opinion_linkage,
                index_in_linked=index_in_linked,
                idle_mode=idle_mode)

            for row in rows_it:
                yield row

    # endregion

    # region private methods

    @staticmethod
    def __create_row_ids_provider(label_provider):
        if isinstance(label_provider, BinaryLabelProvider):
            return BinaryIDProvider()
        if isinstance(label_provider, MultipleLabelProvider):
            return MultipleIDProvider()

    @staticmethod
    def __create_instances_provider(label_provider):
        if isinstance(label_provider, BinaryLabelProvider):
            return MultipleLinkedTextOpinionsInstancesProvider(label_provider.SupportedLabels)
        if isinstance(label_provider, MultipleLabelProvider):
            return SingleInstanceTextOpinionsLinkageProvider()

    def __provide_rows(self, row_dict, parsed_news, text_opinion_linkage, index_in_linked, idle_mode):
        """
        Providing Rows depending on row_id_formatter type
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(row_dict, OrderedDict))
        assert(isinstance(text_opinion_linkage, TextOpinionsLinkage))

        etalon_label = self.__instances_provider.provide_label(text_opinion_linkage)
        for instance in self.__instances_provider.iter_instances(text_opinion_linkage):
            yield self.__create_row(row=row_dict,
                                    parsed_news=parsed_news,
                                    text_opinions_linkage=instance,
                                    index_in_linked=index_in_linked,
                                    # TODO. provide uint_label
                                    etalon_label=etalon_label,
                                    idle_mode=idle_mode)

    def __create_row(self, row, parsed_news, text_opinions_linkage, index_in_linked, etalon_label, idle_mode):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(row, OrderedDict))
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(etalon_label, Label))
        assert(isinstance(idle_mode, bool))

        if idle_mode:
            return None

        text_opinion = text_opinions_linkage[index_in_linked]

        s_ind, t_ind = self.__get_opinion_end_indices(parsed_news, text_opinion)

        row.clear()

        self._fill_row_core(row=row,
                            parsed_news=parsed_news,
                            sentence_ind=TextOpinionHelper.extract_entity_position(
                                parsed_news=parsed_news,
                                text_opinion=text_opinion,
                                end_type=EntityEndType.Source,
                                position_type=TermPositionTypes.SentenceIndex),
                            text_opinion_linkage=text_opinions_linkage,
                            index_in_linked=index_in_linked,
                            etalon_label=etalon_label,
                            s_ind=s_ind,
                            t_ind=t_ind)
        return row

    @staticmethod
    def __get_opinion_end_indices(parsed_news, text_opinion):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind = parsed_news.get_entity_position(text_opinion.SourceId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        t_ind = parsed_news.get_entity_position(text_opinion.TargetId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        return (s_ind, t_ind)

    # endregion

    def set_store_labels(self, store_labels):
        assert(isinstance(store_labels, bool))
        assert(self.__store_labels is None)
        self.__store_labels = store_labels
