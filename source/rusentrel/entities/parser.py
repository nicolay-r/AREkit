from arekit.common.news.entities_parser import BaseEntitiesParser
from arekit.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelTextEntitiesParser(BaseEntitiesParser):

    def parse(self, sentence):
        assert(isinstance(sentence, RuSentRelSentence))

        string_iter = BaseEntitiesParser.iter_text_with_substitutions(
            text=sentence.Text,
            iter_subs=sentence.iter_entity_with_local_bounds())

        return list(string_iter)
