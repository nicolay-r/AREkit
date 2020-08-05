from arekit.common.entities.base import Entity
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.label.base import LabelProvider
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.contrib.networks.features.frames import FrameFeatures
from arekit.networks.input import const


class NetworkSampleFormatter(BaseSampleFormatter):
    """
    Provides additional features, frame-based especially
    """

    def __init__(self, data_type, label_provider, text_provider, synonyms_collection, frames_collection, balance):
        assert(isinstance(label_provider, LabelProvider))
        super(NetworkSampleFormatter, self).__init__(data_type=data_type,
                                                     label_provider=label_provider,
                                                     text_provider=text_provider,
                                                     balance=balance)

        self.__synonyms_collection = synonyms_collection
        self.__frames_collection = frames_collection

    def _get_columns_list_with_types(self):
        dtypes_list = super(NetworkSampleFormatter, self)._get_columns_list_with_types()

        # insert indices
        dtypes_list.append((const.FrameVariantIndices, unicode))
        dtypes_list.append((const.FrameRoles, unicode))
        dtypes_list.append((const.SynonymObject, unicode))
        dtypes_list.append((const.SynonymSubject, unicode))

        return dtypes_list

    def _fill_row_core(self, row, opinion_provider, linked_wrap, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(parsed_news, ParsedNews))

        super(NetworkSampleFormatter, self)._fill_row_core(
            row=row,
            opinion_provider=opinion_provider,
            linked_wrap=linked_wrap,
            index_in_linked=index_in_linked,
            etalon_label=etalon_label,
            parsed_news=parsed_news,
            sentence_ind=sentence_ind,
            s_ind=s_ind, t_ind=t_ind)

        # Extracting list of terms, utilized in further.
        terms = list(self._iter_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind))

        # Compose frame indices.
        uint_frame_inds = list(self.__iter_indices(terms=terms, filter=lambda t: isinstance(t, TextFrameVariant)))

        # Compose frame sentiment.
        uint_frame_roles = FrameFeatures.compose_frame_roles(frame_variants=[terms[fi] for fi in uint_frame_inds],
                                                             frames_collection=self.__frames_collection,
                                                             label_scaler=self._label_provider.LabelScaler)

        # Synonyms for source.
        assert(isinstance(terms[s_ind], Entity))
        syn_s_group = self.__get_g(terms[s_ind].Value)
        uint_syn_s_inds = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(t=t, g=syn_s_group))

        # Synonyms for target.
        assert(isinstance(terms[t_ind], Entity))
        syn_t_group = self.__get_g(terms[t_ind].Value)
        uint_syn_t_inds = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(t=t, g=syn_t_group))

        # Saving.
        row[const.FrameVariantIndices] = self.__to_arg(uint_frame_inds)
        row[const.FrameRoles] = self.__to_arg(uint_frame_roles)
        row[const.SynonymSubject] = self.__to_arg(uint_syn_s_inds)
        row[const.SynonymObject] = self.__to_arg(uint_syn_t_inds)

    @staticmethod
    def __iter_indices(terms, filter):
        for t_ind, term in enumerate(terms):
            if filter(term):
                yield t_ind

    def __syn_check(self, t, g):
        if not isinstance(t, Entity):
            return False
        return self.__get_g(t.Value) == g

    def __get_g(self, value):
        return self.__synonyms_collection.get_synonym_group_index(value)

    def __to_arg(self, inds_iter):
        return const.ArgsSep.join([str(i) for i in inds_iter])
