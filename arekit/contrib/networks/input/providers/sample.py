import collections

from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.contrib.networks.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.input import const
from arekit.contrib.networks.input.providers.term_connotation import extract_uint_frame_variant_connotation
from arekit.contrib.networks.input.rows_parser import create_nn_val_writer_fmt


class NetworkSampleRowProvider(BaseSampleRowProvider):

    def __init__(self,
                 label_provider,
                 text_provider,
                 frames_connotation_provider,
                 frame_role_label_scaler,
                 term_embedding_pairs=None,
                 pos_terms_mapper=None):
        """ term_embedding_pairs: dict or None
                additional structure, utilized to collect all the embedding pairs during the
                rows providing stage.
        """
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(frame_role_label_scaler, SentimentLabelScaler))
        assert(isinstance(pos_terms_mapper, PosTermsMapper) or pos_terms_mapper is None)
        assert(isinstance(term_embedding_pairs, collections.OrderedDict) or term_embedding_pairs is None)

        super(NetworkSampleRowProvider, self).__init__(label_provider=label_provider,
                                                       text_provider=text_provider)

        self.__frames_connotation_provider = frames_connotation_provider
        self.__frame_role_label_scaler = frame_role_label_scaler
        self.__pos_terms_mapper = pos_terms_mapper
        self.__term_embedding_pairs = term_embedding_pairs
        self.__nn_val_fmt = create_nn_val_writer_fmt(fmt_type="writer")

    @property
    def HasEmbeddingPairs(self):
        return self.__term_embedding_pairs is not None

    def iter_term_embedding_pairs(self):
        """ Provide the contents of the embedded pairs.
        """
        return iter(self.__term_embedding_pairs.items())

    def clear_embedding_pairs(self):
        """ Release the contents of the collected embedding pairs.
        """

        # Check whether we actually collect embedding pairs.
        if self.__term_embedding_pairs is None:
            return

        self.__term_embedding_pairs.clear()

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_doc, sentence_ind, s_ind, t_ind):
        assert(isinstance(parsed_doc, ParsedDocument))

        super(NetworkSampleRowProvider, self)._fill_row_core(
            row=row,
            text_opinion_linkage=text_opinion_linkage,
            index_in_linked=index_in_linked,
            etalon_label=etalon_label,
            parsed_doc=parsed_doc,
            sentence_ind=sentence_ind,
            s_ind=s_ind, t_ind=t_ind)

        # Extracting list of terms, utilized in further.
        terms_iter, actual_s_ind, actual_t_ind = self._provide_sentence_terms(
            parsed_doc=parsed_doc, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)
        terms = list(terms_iter)

        # Compose frame indices.
        uint_frame_inds = list(self.__iter_indices(terms=terms, filter=lambda t: isinstance(t, TextFrameVariant)))

        # Compose frame connotations.
        uint_frame_connotations = list(
            map(lambda variant: extract_uint_frame_variant_connotation(
                    text_frame_variant=variant,
                    frames_connotation_provider=self.__frames_connotation_provider,
                    three_label_scaler=self.__frame_role_label_scaler),
                [terms[frame_ind] for frame_ind in uint_frame_inds]))

        vm = {
            const.FrameVariantIndices: uint_frame_inds,
            const.FrameConnotations: uint_frame_connotations,
            const.SynonymSubject: self.__create_synonyms_set(terms=terms, term_ind=actual_s_ind),
            const.SynonymObject: self.__create_synonyms_set(terms=terms, term_ind=actual_t_ind),
            const.PosTags: None if self.__pos_terms_mapper is None else [int(pos_tag) for pos_tag in self.__pos_terms_mapper.iter_mapped(terms)]
        }

        self._apply_row_data(row=row, vm=vm, val_fmt=self.__nn_val_fmt)

    # region private methods

    def __create_synonyms_set(self, terms, term_ind):
        entity = terms[term_ind]
        assert(isinstance(entity, Entity))

        # Searching for other synonyms among all the terms.
        group_ind = entity.GroupIndex
        it = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(term=t, group_ind=group_ind))
        inds_set = set(it)

        # Guarantee the presence of the term_ind
        inds_set.add(term_ind)

        return sorted(inds_set)

    @staticmethod
    def __iter_indices(terms, filter):
        for t_ind, term in enumerate(terms):
            if filter(term):
                yield t_ind

    def __syn_check(self, term, group_ind):
        if not isinstance(term, Entity):
            return False
        if group_ind is None:
            return False
        return term.GroupIndex == group_ind

    # endregion
