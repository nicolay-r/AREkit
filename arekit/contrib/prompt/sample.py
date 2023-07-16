from arekit.common.data import const
from arekit.common.data.input.providers.sample.cropped import CroppedSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter


class PromptedSampleRowProvider(CroppedSampleRowProvider):
    """ Sample, enriched with the prompt technique.
    """

    def __init__(self, crop_window_size, label_scaler, text_provider, prompt, label_fmt=None):
        """ crop_window_size: int
                crop window size for the original text.
            prompt: str
                text which wraps the original cropped (optionally text).
                this string suppose to include the following parameters (optional):
                    text, s_ind, t_ind, s_val, t_val, label_uint
        """
        assert(isinstance(prompt, str))
        assert(isinstance(text_provider, BaseSingleTextProvider))
        assert(isinstance(label_fmt, StringLabelsFormatter) or label_fmt is None)

        super(PromptedSampleRowProvider, self).__init__(crop_window_size=crop_window_size,
                                                        label_scaler=label_scaler,
                                                        text_provider=text_provider)

        self.__prompt = prompt
        self.__labels_fmt = label_fmt

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):

        super(PromptedSampleRowProvider, self)._fill_row_core(row=row,
                                                              text_opinion_linkage=text_opinion_linkage,
                                                              index_in_linked=index_in_linked,
                                                              etalon_label=etalon_label,
                                                              parsed_news=parsed_news,
                                                              sentence_ind=sentence_ind,
                                                              s_ind=s_ind,
                                                              t_ind=t_ind)
        original_text = row[BaseSingleTextProvider.TEXT_A]

        sentence_terms, actual_s_ind, actual_t_ind = self._provide_sentence_terms(
            parsed_news=parsed_news, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)

        label_uint = row[const.LABEL] if const.LABEL in row else None
        label_val = str(label_uint) if label_uint is None or self.__labels_fmt is None else \
            self.__labels_fmt.label_to_str(self._label_provider.LabelScaler.uint_to_label(row[const.LABEL]))

        row[BaseSingleTextProvider.TEXT_A] = self.__prompt.format(
            text=original_text,
            s_ind=row[const.S_IND],
            t_ind=row[const.T_IND],
            s_val=sentence_terms[actual_s_ind].DisplayValue,
            t_val=sentence_terms[actual_t_ind].DisplayValue,
            label_uint=label_uint,
            label_val=label_val)

        return row
