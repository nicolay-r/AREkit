from arekit.common.embeddings.base import Embedding
from arekit.networks.tf_models.single.embedding.custom import create_term_embedding
from arekit.networks.embedding import EntityMasks


# region private functions

def __custom_embedding_func(term, entity_embeddings, word_embedding):
    assert(isinstance(entity_embeddings, dict))
    assert(isinstance(term, unicode))

    if term in entity_embeddings:
        return entity_embeddings[term]

    return create_term_embedding(term=term,
                                 embedding=word_embedding,
                                 max_part_size=3)


def __iter_custom_words(iter_all_terms_func, config):
    assert(callable(iter_all_terms_func))

    all_terms_iter = iter_all_terms_func(lambda t:
                                         isinstance(t, unicode) and
                                         t not in config.WordEmbedding)

    for e_mask in EntityMasks.iter_supported_entity_masks():
        yield EntityMasks.compose(e_mask=e_mask, e_type=None)

    for term in all_terms_iter:
        yield term


# endregion

def init_custom_words_embedding(iter_all_terms_func,
                                entity_embeddings,
                                word_embedding,
                                config):
    assert(isinstance(entity_embeddings, dict))
    assert(callable(iter_all_terms_func))

    return Embedding.from_list_with_embedding_func(
            words_iter=__iter_custom_words(iter_all_terms_func=iter_all_terms_func,
                                           config=config),
            embedding_func=lambda term: __custom_embedding_func(term=term,
                                                                entity_embeddings=entity_embeddings,
                                                                word_embedding=word_embedding))
