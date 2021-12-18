from arekit.common.entities.base import Entity
from arekit.common.labels.base import Label
from arekit.common.news.parsed.base import ParsedNews


class BasePairProvider(object):

    def __init__(self, parsed_news):
        assert(isinstance(parsed_news, ParsedNews))
        self.__entities = parsed_news.iter_entities()

    def _create_pair(self, source_entity, target_entity, label):
        raise NotImplementedError()

    # region private methods

    def _iter_from_entities(self, source_entities, target_entities, label, filter_func=None):
        assert(isinstance(label, Label))
        assert(callable(filter_func) or filter_func is None)

        for source_entity in source_entities:
            for target_entity in target_entities:
                assert (isinstance(source_entity, Entity))
                assert (isinstance(target_entity, Entity))

                if filter_func is not None and not filter_func:
                    continue

                yield self._create_pair(source_entity=source_entity,
                                        target_entity=target_entity,
                                        label=label)

    # endregion

    def iter_from_all(self, label, filter_func):
        assert(isinstance(label, Label))

        return self._iter_from_entities(source_entities=self.__entities,
                                        target_entities=self.__entities,
                                        label=label,
                                        filter_func=filter_func)
