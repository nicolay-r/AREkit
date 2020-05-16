#!/usr/bin/python
import logging

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.ruattitudes.collection import RuAttitudesCollection


stemmer = MystemWrapper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# iterating through collection
for news in RuAttitudesCollection.iter_news(stemmer):
    logger.debug(u"News: {}".format(news.ID))
    for sentence in news.iter_sentences():
        # text
        logger.debug(u" ".join(sentence.ParsedText.Terms).encode('utf-8'))
        # objects
        logger.debug(u",".join([object.get_value() for object in sentence.iter_objects()]))
        # attitudes
        for ref_opinion in sentence.iter_ref_opinions():
            src, target = sentence.get_objects(ref_opinion)
            s = u"{src}->{target} ({label}) (t:[{src_type},{target_type}])".format(
                src=src.get_value(),
                target=target.get_value(),
                label=str(ref_opinion.Sentiment.to_class_str()),
                src_type=src.Type,
                target_type=target.Type).encode('utf-8')
            logger.debug(s)
