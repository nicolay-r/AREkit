from collections import OrderedDict

from arekit.common.data import const
from arekit.common.data.input.providers.instances.multiple import MultipleInstancesLinkedTextOpinionsProvider
from arekit.common.data.input.providers.instances.single import SingleInstanceLinkedDataProvider
from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.label.binary import BinaryLabelProvider
from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.entities.base import Entity
from arekit.common.labels.base import Label

from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.entity_service import EntityEndType, EntityServiceProvider
from arekit.common.docs.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion


# TODO. This is actually a text-opinion related sampler.
# TODO. Here we may expose all the text-opinion related params.
# TODO. With more generalized API in base class.
class BaseSampleRowProvider(BaseRowProvider):
    """ Rows provider for samples storage.
    """

    def __init__(self, label_provider, text_provider):
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(text_provider, BaseSingleTextProvider))
        super(BaseSampleRowProvider, self).__init__()

        self._label_provider = label_provider
        self.__text_provider = text_provider
        self.__instances_provider = self.__create_instances_provider(label_provider)
        self.__store_labels = None
        self._val_fmt = create_base_column_fmt(fmt_type="writer")

    # region properties

    @property
    def LabelProvider(self):
        return self._label_provider

    @property
    def TextProvider(self):
        return self.__text_provider

    # endregion

    # region protected methods

    def _provide_sentence_terms(self, parsed_doc, sentence_ind, s_ind, t_ind):
        terms_iter = parsed_doc.iter_sentence_terms(sentence_index=sentence_ind, return_id=False)
        return list(terms_iter), s_ind, t_ind

    # TODO. This is a very task-specific description, too many data provided.
    # TODO. Switch this API to dict of params
    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_doc, sentence_ind, s_ind, t_ind):
        assert(isinstance(self.__store_labels, bool))

        sentence_terms, actual_s_ind, actual_t_ind = self._provide_sentence_terms(
            parsed_doc=parsed_doc, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)

        # Entity indices from the related context.
        entities = list(filter(lambda term: isinstance(term, Entity), sentence_terms))

        # Values mapping.
        vm = {
            const.ID: self._count_row(),
            const.OPINION_ID: text_opinion_linkage.First.TextOpinionID,
            const.OPINION_LINKAGE_ID: index_in_linked,
            const.DOC_ID: text_opinion_linkage.First.DocID,
            const.SENT_IND: sentence_ind,
            const.ENTITY_VALUES: entities,
            const.ENTITY_TYPES: entities,
            const.ENTITIES: [str(i) for i, t in enumerate(sentence_terms) if isinstance(t, Entity)],
            const.S_IND: actual_s_ind,
            const.T_IND: actual_t_ind,
            const.LABEL_UINT: None,
            const.LABEL_STR: None
        }

        # Compose text value.
        def __assign_value(column, value):
            vm[column] = value

        expected_label = text_opinion_linkage.get_linked_label()

        self.__text_provider.add_text_in_row(
            set_text_func=__assign_value, sentence_terms=sentence_terms,
            s_ind=actual_s_ind, t_ind=actual_t_ind,
            expected_label=expected_label)

        if self.__store_labels:
            l2i = self._label_provider.LabelScaler.label_to_uint
            ui2l = self._label_provider.LabelScaler.uint_to_label
            uint_label = self._label_provider.calculate_output_uint_label(
                expected_uint_label=l2i(expected_label), etalon_uint_label=l2i(etalon_label))
            vm[const.LABEL_UINT] = uint_label
            vm[const.LABEL_STR] = type(ui2l(uint_label)).__name__

        self._apply_row_data(row=row, vm=vm, val_fmt=self._val_fmt)

    @staticmethod
    def _apply_row_data(row, vm, val_fmt):
        for k, v in vm.items():
            if v is None:
                continue
            row[k] = v if k not in val_fmt else val_fmt[k](v)

    def _provide_rows(self, parsed_doc, entity_service, text_opinion_linkage, idle_mode):
        assert(isinstance(idle_mode, bool))

        row_dict = OrderedDict()

        for index_in_linked in range(len(text_opinion_linkage)):

            rows_it = self.__provide_rows(
                parsed_doc=parsed_doc,
                entity_service=entity_service,
                row_dict=row_dict,
                text_opinion_linkage=text_opinion_linkage,
                index_in_linked=index_in_linked,
                idle_mode=idle_mode)

            for row in rows_it:
                yield row

    # endregion

    # region private methods

    @staticmethod
    def __create_instances_provider(label_provider):
        # TODO. #473 related: these label providers are based on text opinion extraction task!
        if isinstance(label_provider, BinaryLabelProvider):
            return MultipleInstancesLinkedTextOpinionsProvider(label_provider.SupportedLabels)
        if isinstance(label_provider, MultipleLabelProvider):
            return SingleInstanceLinkedDataProvider()

    def __provide_rows(self, row_dict, parsed_doc, entity_service,
                       text_opinion_linkage, index_in_linked, idle_mode):
        """
        Providing Rows depending on row_id_formatter type
        """
        assert(isinstance(parsed_doc, ParsedDocument))
        assert(isinstance(row_dict, OrderedDict))
        assert(isinstance(text_opinion_linkage, TextOpinionsLinkage))

        etalon_label = self.__instances_provider.provide_label(text_opinion_linkage)
        for instance in self.__instances_provider.iter_instances(text_opinion_linkage):
            yield self.__create_row(row=row_dict,
                                    row_id=0,
                                    parsed_doc=parsed_doc,
                                    entity_service=entity_service,
                                    text_opinions_linkage=instance,
                                    index_in_linked=index_in_linked,
                                    # TODO. provide uint_label
                                    etalon_label=etalon_label,
                                    idle_mode=idle_mode)

    def __create_row(self, row, row_id, parsed_doc, entity_service, text_opinions_linkage,
                     index_in_linked, etalon_label, idle_mode):
        """
        Composing row in following format:
            [id, label, type, text_a]

        returns: OrderedDict
            row with key values
        """
        assert(isinstance(row, OrderedDict))
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(etalon_label, Label))
        assert(isinstance(idle_mode, bool))

        if idle_mode:
            return None

        text_opinion = text_opinions_linkage[index_in_linked]

        s_ind, t_ind = self.__get_opinion_end_indices(entity_service, text_opinion)

        row.clear()

        source_s_ind = entity_service.extract_entity_position(
            text_opinion=text_opinion, end_type=EntityEndType.Source,
            position_type=TermPositionTypes.SentenceIndex)

        target_s_ind = entity_service.extract_entity_position(
            text_opinion=text_opinion, end_type=EntityEndType.Target,
            position_type=TermPositionTypes.SentenceIndex)

        if target_s_ind != source_s_ind:
            raise Exception("Limitation: Multi-Sentence text_opinions are not supported.")

        self._fill_row_core(row=row,
                            parsed_doc=parsed_doc,
                            sentence_ind=source_s_ind,
                            text_opinion_linkage=text_opinions_linkage,
                            index_in_linked=index_in_linked,
                            etalon_label=etalon_label,
                            s_ind=s_ind,
                            t_ind=t_ind)
        return row

    @staticmethod
    def __get_opinion_end_indices(service, text_opinion):
        assert(isinstance(service, EntityServiceProvider))
        assert(isinstance(text_opinion, TextOpinion))

        s_ind = service.get_entity_position(text_opinion.SourceId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        t_ind = service.get_entity_position(text_opinion.TargetId).get_index(
            position_type=TermPositionTypes.IndexInSentence)

        return s_ind, t_ind

    # endregion

    def set_store_labels(self, store_labels):
        assert(isinstance(store_labels, bool))
        self.__store_labels = store_labels
