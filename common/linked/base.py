from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.text_opinion import TextOpinion


def is_context_continued(text_opinion_helper, cur_opinion, prev_opinion):
    assert(isinstance(text_opinion_helper, TextOpinionHelper))
    assert(isinstance(cur_opinion, TextOpinion))
    assert(isinstance(prev_opinion, TextOpinion))

    end_type = EntityEndType.Source

    s_ind1 = text_opinion_helper.extract_entity_position(
        text_opinion=prev_opinion,
        end_type=end_type,
        position_type=TermPositionTypes.SentenceIndex)

    s_ind2 = text_opinion_helper.extract_entity_position(
        text_opinion=cur_opinion,
        end_type=end_type,
        position_type=TermPositionTypes.SentenceIndex)

    return s_ind1 + 1 == s_ind2


