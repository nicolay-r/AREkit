from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.news.parsed.base import ParsedNews
from arekit.contrib.networks.core.input import const
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.features.term_frame_roles import FrameRoleFeatures


class NetworkSampleRowProvider(BaseSampleRowProvider):

    def __init__(self,
                 label_provider,
                 text_provider,
                 frames_connotation_provider,
                 entity_to_group_func,
                 frame_role_label_scaler,
                 pos_terms_mapper):
        assert(isinstance(label_provider, LabelProvider))
        assert(isinstance(pos_terms_mapper, PosTermsMapper))
        assert(isinstance(frame_role_label_scaler, BaseLabelScaler))
        assert(callable(entity_to_group_func))

        super(NetworkSampleRowProvider, self).__init__(label_provider=label_provider,
                                                       text_provider=text_provider)

        self.__entity_to_group_func = entity_to_group_func
        self.__frames_connotation_provider = frames_connotation_provider
        self.__frame_role_label_scaler = frame_role_label_scaler
        self.__pos_terms_mapper = pos_terms_mapper

    def _fill_row_core(self, row, linked_wrap, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(parsed_news, ParsedNews))

        super(NetworkSampleRowProvider, self)._fill_row_core(
            row=row,
            linked_wrap=linked_wrap,
            index_in_linked=index_in_linked,
            etalon_label=etalon_label,
            parsed_news=parsed_news,
            sentence_ind=sentence_ind,
            s_ind=s_ind, t_ind=t_ind)

        # Extracting list of terms, utilized in further.
        terms = list(self._provide_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind))

        # Compose frame indices.
        uint_frame_inds = list(self.__iter_indices(terms=terms, filter=lambda t: isinstance(t, TextFrameVariant)))

        # Compose frame sentiment.
        uint_frame_roles = list(
            map(lambda variant: FrameRoleFeatures.extract_uint_frame_variant_sentiment_role(
                    text_frame_variant=variant,
                    frames_connotation_provider=self.__frames_collection,
                    three_label_scaler=self.__frame_role_label_scaler),
                [terms[frame_ind] for frame_ind in uint_frame_inds]))

        # Synonyms for source.
        uint_syn_s_inds = self.__create_synonyms_set(terms=terms, term_ind=s_ind)

        # Synonyms for target.
        uint_syn_t_inds = self.__create_synonyms_set(terms=terms, term_ind=t_ind)

        # Entity indicies from the related context.
        entity_inds = list(self.__iter_indices(terms=terms, filter=lambda t: self.__is_entity(t)))

        # Part of speech tags
        pos_int_tags = [int(pos_tag) for pos_tag in self.__pos_terms_mapper.iter_mapped(terms)]

        # Saving.
        row[const.FrameVariantIndices] = self.__to_arg(uint_frame_inds)
        row[const.FrameRoles] = self.__to_arg(uint_frame_roles)
        row[const.SynonymSubject] = self.__to_arg(uint_syn_s_inds)
        row[const.SynonymObject] = self.__to_arg(uint_syn_t_inds)
        row[const.Entities] = self.__to_arg(entity_inds)
        row[const.PosTags] = self.__to_arg(pos_int_tags)

    # region private methods

    @staticmethod
    def __is_entity(t):
        return isinstance(t, Entity)

    def __create_synonyms_set(self, terms, term_ind):
        entity = terms[term_ind]
        assert(isinstance(entity, Entity))

        # Searching for other synonyms among all the terms.
        group_ind = self.__entity_to_group_func(entity)
        it = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(term=t, group_ind=group_ind))
        inds_set = set(it)

        # Guarantee the presence of the term_ind
        inds_set.add(term_ind)

        return sorted(inds_set)

    @staticmethod
    def __iter_indices(terms, filter):
        for t_ind, term in enumerate(terms):
            if list(filter(term)):
                yield t_ind

    def __syn_check(self, term, group_ind):
        if not isinstance(term, Entity):
            return False
        if group_ind is None:
            return False
        return self.__entity_to_group_func(term) == group_ind

    @staticmethod
    def __to_arg(inds_iter):
        return const.ArgsSep.join([str(i) for i in inds_iter])

    # endregion
