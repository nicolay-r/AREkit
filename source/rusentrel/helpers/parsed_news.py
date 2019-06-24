from core.common.parsed_news.parsed_news import ParsedNews
from core.processing.text.parser import TextParser
from core.source.rusentrel.entities.entity import RuSentRelEntity
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelParsedNews(ParsedNews):

    @classmethod
    def create_from_rusentrel_news(cls, rusentrel_news_id, rusentrel_news, keep_tokens):
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(keep_tokens, bool))

        terms, entity_positions, sentence_begin_inds = cls.__process(
            rusentrel_news,
            keep_tokens)

        return cls(news_id=rusentrel_news_id,
                   terms=terms,
                   entity_positions=entity_positions,
                   sentence_begin_inds=sentence_begin_inds)

    # TODO. Simplify
    @staticmethod
    def __process(rusentrel_news, keep_tokens):
        assert(isinstance(rusentrel_news, RuSentRelNews))
        assert(isinstance(keep_tokens, bool))

        sentence_begin = []
        terms = []
        entity_positions = {}
        for s_index, sentence in enumerate(rusentrel_news.iter_sentences()):
            assert(isinstance(sentence, RuSentRelSentence))
            sentence_begin.append(len(terms))
            s_pos = 0

            # TODO: guarantee that entities ordered by e_begin.
            for e in sentence.iter_entities():
                assert(isinstance(e, RuSentRelEntity))
                # add parsed_news before entity
                if e.CharIndexBegin > s_pos:
                    parsed_text_before = TextParser.parse(text=sentence.Text[s_pos:e.CharIndexBegin],
                                                          keep_tokens=keep_tokens)
                    terms.extend(parsed_text_before.iter_raw_terms())

                # add entity position
                entity_positions[e.IdInDocument] = (len(terms), s_index)
                # add entity_text
                terms.append(rusentrel_news.DocEntities.get_entity_by_id(e.IdInDocument))
                s_pos = e.CharIndexEnd

            # add text part after last entity of sentence.
            parsed_text_last = TextParser.parse(text=sentence.Text[s_pos:len(sentence.Text)],
                                                keep_tokens=keep_tokens)
            terms.extend(parsed_text_last.iter_raw_terms())

        return terms, entity_positions, sentence_begin
