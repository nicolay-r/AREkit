from arekit.common.embeddings.base import Embedding
from arekit.common.entities.str_fmt import StringEntitiesFormatter


def generate_entity_embeddings(word_embedding,
                               string_entity_formatter,
                               string_emb_entity_formatter):
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(string_entity_formatter, StringEntitiesFormatter))
    assert(isinstance(string_emb_entity_formatter, StringEntitiesFormatter))

    # Unique start index
    embeddings = {}

    for e_type in string_entity_formatter.iter_supported_types():
        value = string_entity_formatter.to_string(original_value=None,
                                                  entity_type=e_type)

        if value in embeddings:
            continue

        word = string_emb_entity_formatter.to_string(original_value=None,
                                                     entity_type=e_type)
        m_ind = word_embedding.try_find_index_by_plain_word(word)
        embeddings[value] = word_embedding.get_vector_by_index(m_ind)

    return embeddings

