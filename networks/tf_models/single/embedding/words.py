from arekit.common.embeddings.base import Embedding
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.networks.tf_models.single.embedding.custom import create_term_embedding


# region private functions

def __custom_embedding_func(term, entity_embeddings, word_embedding):
    assert(isinstance(entity_embeddings, dict))
    assert(isinstance(term, unicode))

    if term in entity_embeddings:
        return entity_embeddings[term]

    return create_term_embedding(term=term,
                                 embedding=word_embedding,
                                 max_part_size=3)


def __iter_custom_words(iter_all_terms_func,
                        string_entity_formatter,
                        config):
    assert(isinstance(string_entity_formatter, StringEntitiesFormatter))
    assert(callable(iter_all_terms_func))

    all_terms_iter = iter_all_terms_func(lambda t:
                                         isinstance(t, unicode) and
                                         t not in config.WordEmbedding)

    for entity_type in string_entity_formatter.iter_supported_types():
        yield string_entity_formatter.to_string(entity_type=entity_type,
                                                original_value=None)

    for term in all_terms_iter:
        yield term


# endregion

def init_custom_words_embedding(iter_all_terms_func,
                                entity_embeddings,
                                word_embedding,
                                string_entity_formatter,
                                config):
    assert(isinstance(entity_embeddings, dict))
    assert(isinstance(string_entity_formatter, StringEntitiesFormatter))
    assert(callable(iter_all_terms_func))

    return Embedding.from_list_with_embedding_func(
            words_iter=__iter_custom_words(iter_all_terms_func=iter_all_terms_func,
                                           string_entity_formatter=string_entity_formatter,
                                           config=config),
            embedding_func=lambda term: __custom_embedding_func(term=term,
                                                                entity_embeddings=entity_embeddings,
                                                                word_embedding=word_embedding))
