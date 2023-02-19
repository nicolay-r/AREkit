from arekit.common.data import const
from arekit.common.data.input.providers.sample.cropped import CroppedSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider


class PromptedSampleRowProvider(CroppedSampleRowProvider):
    """ Sample, enriched with the prompt technique.
    """

    def __init__(self, crop_window_size, label_scaler, text_provider, prompt):
        """ crop_window_size: int
                crop window size for the original text.
            prompt: str
                text which wraps the original cropped (optionally text).
                this string suppose to include the following parameters:
                    text, s_ind, r_ind, label (optional)
        """
        assert(isinstance(prompt, str))
        assert(isinstance(text_provider, BaseSingleTextProvider))

        super(PromptedSampleRowProvider, self).__init__(crop_window_size=crop_window_size,
                                                        label_scaler=label_scaler,
                                                        text_provider=text_provider)

        self.__prompt = prompt

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):

        row = super(PromptedSampleRowProvider, self)._fill_row_core(row=row,
                                                                    text_opinion_linkage=text_opinion_linkage,
                                                                    index_in_linked=index_in_linked,
                                                                    etalon_label=etalon_label,
                                                                    parsed_news=parsed_news,
                                                                    sentence_ind=sentence_ind,
                                                                    s_ind=s_ind,
                                                                    t_ind=t_ind)
        original_text = row[BaseSingleTextProvider.TEXT_A]
        row[BaseSingleTextProvider.TEXT_A] = self.__prompt.format(
            text=original_text,
            s_ind=row[const.S_IND],
            t_ind=row[const.T_IND],
            label=row[const.LABEL] if const.LABEL in row else None)

        return row
