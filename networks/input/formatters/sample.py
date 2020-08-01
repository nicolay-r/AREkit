from arekit.common.entities.base import Entity
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_frame_variant import TextFrameVariant


class NetworkSampleFormatter(BaseSampleFormatter):
    """
    Provides additional features, frame-based especially
    """

    Frames = "frames"
    SynonymObject = "syn_obj"
    SynonymSubject = "syn_subj"
    ArgsSep = u','

    def __init__(self, data_type, label_provider, text_provider, synonyms_collection, balance):
        assert(isinstance(synonyms_collection, SynonymsCollection))
        super(NetworkSampleFormatter, self).__init__(data_type=data_type,
                                                     label_provider=label_provider,
                                                     text_provider=text_provider,
                                                     balance=balance)

        self.__synonyms_collection = synonyms_collection

    def _fill_row_core(self, row, opinion_provider, linked_wrap, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):
        assert(isinstance(parsed_news, ParsedNews))

        super(NetworkSampleFormatter, self)._fill_row_core(
            row=row,
            opinion_provider=opinion_provider,
            linked_wrap=linked_wrap,
            index_in_linked=index_in_linked,
            etalon_label=etalon_label,
            parsed_news=parsed_news, sentence_ind=sentence_ind,
            s_ind=s_ind, t_ind=t_ind)

        # Extracting list of terms, utilized in further.
        terms = list(self._iter_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind))

        # Compose frame indices.
        frame_inds = self.__iter_indices(terms=terms, filter=lambda t: isinstance(t, TextFrameVariant))

        # TODO. For frame roles: we have frames and hence information to obtain related sentiment.

        # Synonyms for source.
        assert(terms[s_ind], Entity)
        syn_s_group = self.__get_g(terms[s_ind].Value)
        syn_s_inds = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(t=t, g=syn_s_group))

        # Synonyms for target.
        assert(terms[t_ind], Entity)
        syn_t_group = self.__get_g(terms[t_ind].Value)
        syn_t_inds = self.__iter_indices(terms=terms, filter=lambda t: self.__syn_check(t=t, g=syn_t_group))

        # Saving.
        row[self.Frames] = self.__to_arg(frame_inds)
        row[self.SynonymSubject] = self.__to_arg(syn_s_inds)
        row[self.SynonymObject] = self.__to_arg(syn_t_inds)

    @staticmethod
    def __iter_indices(terms, filter):
        for t_ind, term in enumerate(terms):
            if filter(term):
                yield str(t_ind)

    def __syn_check(self, t, g):
        if not isinstance(t, Entity):
            return False
        return self.__get_g(t.Value) == g

    def __get_g(self, value):
        return self.__synonyms_collection.get_synonym_group_index(value)

    def __to_arg(self, inds_iter):
        return self.ArgsSep.join(inds_iter)

