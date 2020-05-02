from arekit.common import utils
from arekit.common.entities.base import Entity
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.text_object import TextObject
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesParsedNewsHelper(object):

    # region private methods

    @staticmethod
    def __iter_parsed_sentences(news):
        """
        This method returns sentences with labeled entities in it.
        """
        assert(isinstance(news, RuAttitudesNews))

        objects_read = 0
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))

            yield RuAttitudesParsedNewsHelper.__iter_terms_with_entities(
                sentence=sentence,
                s_to_doc_id=lambda s_level_id: s_level_id + objects_read)

            objects_read += sentence.ObjectsCount

    @staticmethod
    def __iter_terms_with_entities(sentence, s_to_doc_id):
        assert(isinstance(sentence, RuAttitudesSentence))
        assert(callable(s_to_doc_id))

        subs_iter = RuAttitudesParsedNewsHelper.__iter_subs(
            sentence=sentence,
            s_to_doc_id=s_to_doc_id)

        terms_with_entities_iter = utils.iter_text_with_substitutions(
            text=list(sentence.ParsedText.iter_raw_terms()),
            iter_subs=subs_iter)

        return sentence.ParsedText.copy_modified(terms=list(terms_with_entities_iter))

    @staticmethod
    def __iter_subs(sentence, s_to_doc_id):
        assert(isinstance(sentence, RuAttitudesSentence))
        assert(callable(s_to_doc_id))

        for s_level_id, obj in enumerate(sentence.iter_objects()):
            assert(isinstance(obj, TextObject))

            _value = obj.get_value()
            value = _value if len(_value) > 0 else u'[empty]'

            entity = Entity(value=value,
                            e_type=obj.Type,
                            id_in_doc=s_to_doc_id(s_level_id))

            yield entity, obj.get_bound()

    # endregion

    # TODO. rename to Parse
    # TODO. rusentrel_news_id -> doc_id (REMOVE, since it will be available in News base class)
    @classmethod
    def create_parsed_news(cls, doc_id, news):
        assert(isinstance(doc_id, int))
        assert(isinstance(news, RuAttitudesNews))

        parsed_sentences_iter = cls.__iter_parsed_sentences(news)

        return ParsedNews(news_id=doc_id,
                          parsed_sentences=parsed_sentences_iter)
