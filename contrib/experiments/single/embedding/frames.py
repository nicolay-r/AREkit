from arekit.common.embeddings.base import Embedding
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.contrib.experiments.single.embedding.custom import create_term_embedding
from arekit.contrib.experiments.single.initialization import SingleInstanceModelInitializer


def __iter_variants(m_init):
    assert(isinstance(m_init, SingleInstanceModelInitializer))
    frame_variants_iter = m_init.iter_all_terms(lambda t: isinstance(t, TextFrameVariant))
    for variant in frame_variants_iter:
        yield variant.Variant.get_value()


def init_frames_embedding(m_init, word_embedding):
    assert(isinstance(m_init, SingleInstanceModelInitializer))
    return Embedding.from_list_with_embedding_func(
        words_iter=__iter_variants(m_init),
        embedding_func=lambda variant_value: create_term_embedding(term=variant_value,
                                                                   embedding=word_embedding,
                                                                   max_part_size=3))

