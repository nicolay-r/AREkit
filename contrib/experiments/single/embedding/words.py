from arekit.common.embeddings.base import Embedding
from arekit.contrib.experiments.single.embedding.entities import iter_all_entity_types
from arekit.contrib.experiments.single.initialization import SingleInstanceModelInitializer
from arekit.networks.context.embedding import entity


def __custom_embedding_func(self, term, word_embedding):
    assert(isinstance(term, unicode))

    if term in self.__entity_embeddings:
        return self.__entity_embeddings[term]

    # TODO. Entity has _ separator!!!
    return word_embedding.create_term_embedding(term)


def __iter_custom_words(m_init, config):
    assert(isinstance(m_init, SingleInstanceModelInitializer))

    all_terms_iter = m_init.iter_all_terms(lambda t:
                                           isinstance(t, unicode) and
                                           t not in config.WordEmbedding)

    for e_mask in entity.iter_entity_masks():
        for e_type in iter_all_entity_types():
            # TODO. Entity has a different separator for type
            yield entity.compose_entity_mask(e_mask=e_mask, e_type=e_type)
            # TODO. Entity has a different separator for type
        yield entity.compose_entity_mask(e_mask=e_mask, e_type=None)

    for term in all_terms_iter:
        yield term


def init_custom_words_embedding(m_init, word_embedding, config):
    assert(isinstance(m_init, SingleInstanceModelInitializer))

    return Embedding.from_list_with_embedding_func(
            words_iter=__iter_custom_words(m_init=m_init, config=config),
            embedding_func=lambda term: __custom_embedding_func(term, word_embedding=word_embedding))



