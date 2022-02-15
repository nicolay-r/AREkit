from arekit.common.labels.provider.base import BasePairLabelProvider
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider


class BasePairProvider(BaseParsedNewsServiceProvider):

    @property
    def Name(self):
        raise NotImplementedError()

    def _create_pair(self, source_entity, target_entity, label):
        raise NotImplementedError()

    # region private methods

    def _iter_from_entities(self, src_entity_doc_ids, tgt_entity_doc_ids, label_provider, filter_func=None):
        assert(isinstance(src_entity_doc_ids, list))
        assert(isinstance(tgt_entity_doc_ids, list))
        assert(isinstance(label_provider, BasePairLabelProvider))
        assert(callable(filter_func) or filter_func is None)

        for src_e_doc_id in src_entity_doc_ids:
            for tgt_e_doc_id in tgt_entity_doc_ids:
                assert(isinstance(src_e_doc_id, int))
                assert(isinstance(tgt_e_doc_id, int))

                # Extract entities by doc_id.
                source_entity = self._doc_entities[src_e_doc_id]
                target_entity = self._doc_entities[tgt_e_doc_id]

                if filter_func is not None and not filter_func(source_entity, target_entity):
                    continue

                if source_entity == target_entity:
                    continue

                label = label_provider.provide(source=source_entity,
                                               target=target_entity)

                yield self._create_pair(source_entity=source_entity,
                                        target_entity=target_entity,
                                        label=label)

    # endregion

    def iter_from_all(self, label_provider, filter_func):
        assert(isinstance(label_provider, BasePairLabelProvider))
        return self._iter_from_entities(src_entity_doc_ids=list(map(lambda e: e.IdInDocument, self._doc_entities)),
                                        tgt_entity_doc_ids=list(map(lambda e: e.IdInDocument, self._doc_entities)),
                                        label_provider=label_provider,
                                        filter_func=filter_func)
