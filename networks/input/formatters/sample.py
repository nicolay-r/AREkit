from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text_frame_variant import TextFrameVariant


class NetworkSampleFormatter(BaseSampleFormatter):
    """
    Provides additional features, frame-based especially
    """

    Frames = "frames"

    # TODO. Provide TOH
    def __init__(self, data_type, label_provider, text_provider, balance):
        super(NetworkSampleFormatter, self).__init__(data_type=data_type,
                                                     label_provider=label_provider,
                                                     text_provider=text_provider,
                                                     balance=balance)

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

        # TODO. Fill row with extra parameters: (REFER TO samples.py)

        # Compose frame indices.
        frame_inds = [unicode(t_ind) for t_ind, term in
                      enumerate(self._iter_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind))
                      if isinstance(term, TextFrameVariant)]
        row[self.Frames] = u",".join(frame_inds)

        # TODO. For frame roles: we have frames and hence information to obtain related sentiment.

        # TODO. For synonyms to subj: find all Entities, which similar to Entity(s_subj)
        # TODO.     Use helper for this example.

        # TODO. For synonyms to obj: find all Entities, which similar to Entity(s_obj)
        # TODO.     Use helper for this example.

