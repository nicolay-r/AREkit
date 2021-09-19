from collections import OrderedDict

from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment import const
from arekit.common.experiment.input.providers.label.base import LabelProvider
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.providers.rows.base import BaseRowProvider
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.bert.core.input.providers.label.binary import BinaryLabelProvider
from arekit.contrib.bert.core.input.providers.row_ids.binary import BinaryIDProvider


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
    def _iter_sentence_terms(parsed_news, sentence_ind):
        return parsed_news.iter_sentence_terms(sentence_index=sentence_ind, return_id=False)

    def _fill_row_core(self, row, linked_wrap, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(self.__store_labels, bool))

        def __assign_value(column, value):
            row[column] = value

        row[const.ID] = self.__row_ids_provider.create_sample_id(
            linked_opinions=linked_wrap,
            index_in_linked=index_in_linked,
            label_scaler=self._label_provider.LabelScaler)

        row[const.NEWS_ID] = linked_wrap.First.NewsID

        expected_label = linked_wrap.get_linked_label()

        if self.__store_labels:
            row[const.LABEL] = self._label_provider.calculate_output_uint_label(
                expected_uint_label=self._label_provider.LabelScaler.label_to_uint(expected_label),
                etalon_uint_label=self._label_provider.LabelScaler.label_to_uint(etalon_label))

        self.__text_provider.add_text_in_row(
            set_text_func=lambda column, value: __assign_value(column, value),
            sentence_terms=list(self._iter_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind)),
            s_ind=s_ind,
            t_ind=t_ind,
            expected_label=expected_label)

        row[const.S_IND] = s_ind
        row[const.T_IND] = t_ind

    def _provide_rows(self, parsed_news, linked_wrapper, idle_mode):
        assert(isinstance(idle_mode, bool))

        row_dict = OrderedDict()

        for index_in_linked in range(len(linked_wrapper)):

            rows_it = self.__provide_rows(
                parsed_news=parsed_news,
                row_dict=row_dict,
                linked_wrap=linked_wrapper,
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

    def __provide_rows(self, row_dict, parsed_news, linked_wrap, index_in_linked, idle_mode):
        """
        Providing Rows depending on row_id_formatter type
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(row_dict, OrderedDict))
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))

        origin = linked_wrap.First
        if isinstance(self.__row_ids_provider, BinaryIDProvider):
            """
            Enumerate all opinions as if it would be with the different label types.
            """
            for label in self._label_provider.SupportedLabels:
                yield self.__create_row(row=row_dict,
                                        parsed_news=parsed_news,
                                        linked_wrap=self.__copy_modified_linked_wrap(linked_wrap, label),
                                        index_in_linked=index_in_linked,
                                        # TODO. provide uint_label
                                        etalon_label=origin.Sentiment,
                                        idle_mode=idle_mode)

        if isinstance(self.__row_ids_provider, MultipleIDProvider):
            yield self.__create_row(row=row_dict,
                                    parsed_news=parsed_news,
                                    linked_wrap=linked_wrap,
                                    index_in_linked=index_in_linked,
                                    # TODO. provide uint_label
                                    etalon_label=origin.Sentiment,
                                    idle_mode=idle_mode)

    def __create_row(self, row, parsed_news, linked_wrap, index_in_linked, etalon_label, idle_mode):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(row, OrderedDict))
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(etalon_label, Label))
        assert(isinstance(idle_mode, bool))

        if idle_mode:
            return None

        text_opinion = linked_wrap[index_in_linked]

        s_ind, t_ind = self.__get_opinion_end_indices(parsed_news, text_opinion)

        row.clear()

        self._fill_row_core(row=row,
                            parsed_news=parsed_news,
                            sentence_ind=TextOpinionHelper.extract_entity_position(
                                parsed_news=parsed_news,
                                text_opinion=text_opinion,
                                end_type=EntityEndType.Source,
                                position_type=TermPositionTypes.SentenceIndex),
                            linked_wrap=linked_wrap,
                            index_in_linked=index_in_linked,
                            etalon_label=etalon_label,
                            s_ind=s_ind,
                            t_ind=t_ind)
        return row

    @staticmethod
    def __copy_modified_linked_wrap(linked_wrap, label):
        assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
        linked_opinions = [o for o in linked_wrap]

        copy = TextOpinion.create_copy(other=linked_opinions[0])
        copy.set_label(label=label)

        linked_opinions[0] = copy

        return LinkedTextOpinionsWrapper(linked_text_opinions=linked_opinions)

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
