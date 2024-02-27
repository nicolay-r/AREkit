import logging

from arekit.common.context.token import Token
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.utils.processing.text.tokens import Tokens

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DefaultTextTokenizer(BasePipelineItem):
    """ Default parser implementation.
    """

    def __init__(self, keep_tokens=True, **kwargs):
        super(DefaultTextTokenizer, self).__init__(**kwargs)
        self.__keep_tokens = keep_tokens

    # region protected methods

    def apply_core(self, input_data, pipeline_ctx):
        output_data = self.__process_parts(input_data)
        if not self.__keep_tokens:
            output_data = [word for word in output_data if not isinstance(word, Token)]
        return output_data

    # endregion

    # region private static methods

    def __process_parts(self, parts):
        assert(isinstance(parts, list))

        parsed = []
        for part in parts:

            if part is None:
                continue

            # Keep non str words as it is and try to parse str-based words.
            processed = [part] if not isinstance(part, str) else \
                self.__iter_processed_part(part=part)

            parsed.extend(processed)

        return parsed

    def __iter_processed_part(self, part):
        for word in split_by_whitespaces(part):
            for term in self.__process_word(word):
                yield term

    def __process_word(self, word):
        assert(isinstance(word, str))
        return self.__split_tokens(word)

    @staticmethod
    def __split_tokens(term):
        """
        Splitting off tokens from parsed_doc ending, i.e. for example:
            term: "сказать,-" -> "(term: "сказать", ["COMMA_TOKEN", "DASH_TOKEN"])
        return: (unicode or None, list)
            modified term and list of extracted tokens.
        """

        url = Tokens.try_create_url(term)
        if url is not None:
            return [url]

        l = 0
        words_and_tokens = []
        while l < len(term):

            # Token.
            token = Tokens.try_create(term[l])
            if token is not None:
                if token.get_token_value() != Tokens.NEW_LINE:
                    words_and_tokens.append(token)
                l += 1

            # Number.
            elif str.isdigit(term[l]):
                k = l + 1
                while k < len(term) and str.isdigit(term[k]):
                    k += 1
                token = Tokens.try_create_number(term[l:k])
                assert(token is not None)
                words_and_tokens.append(token)
                l = k

            # Term.
            else:
                k = l + 1
                while k < len(term):
                    token = Tokens.try_create(term[k])
                    if token is not None and token.get_token_value() != Tokens.DASH:
                        break
                    k += 1
                words_and_tokens.append(term[l:k])
                l = k

        return words_and_tokens

    # endregion
