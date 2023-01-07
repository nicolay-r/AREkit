from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.common.news.parsed.base import ParsedNews
from arekit.contrib.networks.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.input import const
from arekit.contrib.networks.input.providers.term_connotation import extract_uint_frame_variant_connotation


class NetworkSampleRowProvider(BaseSampleRowProvider):

    def __init__(self,
                 label_provider,
                 text_provider,
                 frames_connotation_provider,
                 frame_role_label_scaler,
                 pos_terms_mapper=None):
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(frame_role_label_scaler, SentimentLabelScaler))
        assert(isinstance(pos_terms_mapper, PosTermsMapper) or pos_terms_mapper is None)

        super(NetworkSampleRowProvider, self).__init__(label_provider=label_provider,
                                                       text_provider=text_provider)

        self.__frames_connotation_provider = frames_connotation_provider
        self.__frame_role_label_scaler = frame_role_label_scaler
        self.__pos_terms_mapper = pos_terms_mapper

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(parsed_news, ParsedNews))

        super(NetworkSampleRowProvider, self)._fill_row_core(
            row=row,
            text_opinion_linkage=text_opinion_linkage,
            index_in_linked=index_in_linked,
            etalon_label=etalon_label,
            parsed_news=parsed_news,
            sentence_ind=sentence_ind,
            s_ind=s_ind, t_ind=t_ind)

        # Extracting list of terms, utilized in further.
        terms_iter, actual_s_ind, actual_t_ind = self._provide_sentence_terms(
            parsed_news=parsed_news, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)
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

        # Synonyms for source.
        uint_syn_s_inds = self.__create_synonyms_set(terms=terms, term_ind=actual_s_ind)

        # Synonyms for target.
        uint_syn_t_inds = self.__create_synonyms_set(terms=terms, term_ind=actual_t_ind)

        # Part of speech tags
        pos_int_tags = None if self.__pos_terms_mapper is None \
            else [int(pos_tag) for pos_tag in self.__pos_terms_mapper.iter_mapped(terms)]

        # Saving.
        row[const.FrameVariantIndices] = self.__to_arg(uint_frame_inds)
        row[const.FrameConnotations] = self.__to_arg(uint_frame_connotations)
        row[const.SynonymSubject] = self.__to_arg(uint_syn_s_inds)
        row[const.SynonymObject] = self.__to_arg(uint_syn_t_inds)
        if pos_int_tags is not None:
            row[const.PosTags] = self.__to_arg(pos_int_tags)

    # region private methods

    @staticmethod
    def __is_entity(t):
        return isinstance(t, Entity)

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

    @staticmethod
    def __to_arg(inds_iter):
        return const.ArgsSep.join([str(i) for i in inds_iter])

    # endregion
