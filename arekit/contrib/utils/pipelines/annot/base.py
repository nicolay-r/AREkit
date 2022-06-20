from arekit.common.news.parser import NewsParser
from arekit.common.opinions.annot.base import BaseAnnotator
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.annot.opinion_annotation import ppl_text_ids_to_parsed_news, \
    ppl_parsed_to_annotation, ppl_parsed_news_to_opinion_linkages


def attitude_extraction_default_pipeline(annotator, data_type, get_doc_func, text_parser,
                                         value_to_group_id_func, terms_per_context):
    """ This is a default pipeline which found its application in Sentiment Attitude Extraction task [1].
        In a nutshell, the sequence of processing operations is as follows:

        get_doc_func:
            func(doc_id)

        result: Pipeline with the following transformation
            doc_id -> parsed_news -> annot -> opinion linkages

        References:
            [1] Extracting Sentiment Attitudes from Analytical Texts https://arxiv.org/pdf/1808.08932.pdf
    """
    assert(callable(get_doc_func))
    assert(isinstance(annotator, BaseAnnotator))
    assert(isinstance(text_parser, BaseTextParser))

    return BasePipeline(
        ppl_text_ids_to_parsed_news(
            parse_news_func=lambda doc_id: NewsParser.parse(
                news=get_doc_func(doc_id),
                text_parser=text_parser))
        +
        ppl_parsed_to_annotation(annotator=annotator, data_type=data_type)
        +
        ppl_parsed_news_to_opinion_linkages(value_to_group_id_func=value_to_group_id_func,
                                            terms_per_context=terms_per_context)
    )
