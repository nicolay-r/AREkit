from arekit.common.embeddings.base import Embedding
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.contrib.experiments.single.embedding.custom import create_term_embedding


def __iter_variants(iter_all_terms_func):
    assert(callable(iter_all_terms_func))
    frame_variants_iter = iter_all_terms_func(lambda t: isinstance(t, TextFrameVariant))
    for variant in frame_variants_iter:
        yield variant.Variant.get_value()


def init_frames_embedding(word_embedding, iter_all_terms_func):
    assert(callable(iter_all_terms_func))

    return Embedding.from_list_with_embedding_func(
        words_iter=__iter_variants(iter_all_terms_func),
        embedding_func=lambda variant_value: create_term_embedding(term=variant_value,
                                                                   embedding=word_embedding,
                                                                   max_part_size=3))

