from collections import OrderedDict

from arekit.common.entities.base import Entity
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.labels.base import Label
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.processing.text.token import Token


class SingleTextProvider(object):

    TEXT_A = u'text_a'
    TERMS_SEPARATOR = u" "

    def __init__(self, entities_formatter, synonyms):
        assert(isinstance(entities_formatter, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))

        self._entities_formatter = entities_formatter
        self.__synonyms = synonyms

    def iter_columns(self):
        yield SingleTextProvider.TEXT_A

    # TODO. Future: move to core (since 20.1.5 version)
    def _iterate_sentence_terms(self, sentence_terms, s_ind=None, t_ind=None):

        def __syn_group(index):
            if index is None:
                return None

            entity = sentence_terms[index]
            assert(isinstance(entity, Entity))

            if not self.__synonyms.contains_synonym_value(entity.Value):
                return None

            return self.__synonyms.get_synonym_group_index(entity.Value)

        assert(isinstance(sentence_terms, list))
        assert(isinstance(s_ind, int) or s_ind is None)
        assert(isinstance(t_ind, int) or t_ind is None)

        s_group = __syn_group(s_ind)
        t_group = __syn_group(t_ind)

        for i, term in enumerate(sentence_terms):

            if isinstance(term, unicode):
                assert(i != s_ind and i != t_ind)
                yield term
            elif isinstance(term, Entity):
                if i == s_ind:
                    yield self._entities_formatter.to_string(term, EntityType.Subject)
                elif i == t_ind:
                    yield self._entities_formatter.to_string(term, EntityType.Object)
                elif __syn_group(i) == s_group:
                    yield self._entities_formatter.to_string(term, EntityType.SynonymSubject)
                elif __syn_group(i) == t_group:
                    yield self._entities_formatter.to_string(term, EntityType.SynonymObject)
                else:
                    yield self._entities_formatter.to_string(term, EntityType.Other)
            elif isinstance(term, Token):
                yield term.get_original_value()
            elif isinstance(term, TextFrameVariant):
                yield term.Variant.get_value()
            else:
                raise Exception("Term type is not supported: {}".format(type(term)))

    @staticmethod
    def _process_text(text):
        assert(isinstance(text, unicode))
        return text.strip()

    def add_text_in_row(self, row, sentence_terms, s_ind, t_ind, expected_label):
        assert(isinstance(row, OrderedDict))
        assert(isinstance(sentence_terms, list))
        assert(isinstance(expected_label, Label))
        text = self.TERMS_SEPARATOR.join(self._iterate_sentence_terms(sentence_terms,
                                                                      s_ind=s_ind,
                                                                      t_ind=t_ind))
        row[self.TEXT_A] = self._process_text(text)
