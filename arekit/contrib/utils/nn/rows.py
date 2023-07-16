import collections

from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.contrib.networks.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.input.providers.text import NetworkSingleTextProvider
from arekit.contrib.networks.input.term_types import TermTypes
from arekit.contrib.networks.input.terms_mapping import VectorizedNetworkTermMapping
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer


def __add_term_embedding(dict_data, term, emb_vector):
    if term in dict_data:
        return
    dict_data[term] = emb_vector


def create_rows_provider(str_entity_fmt, ctx, vectorizers="default"):
    """ This method is corresponds to the default initialization of
        the rows provider for data sampling pipeline.

        vectorizers:
            NONE: no need to vectorize, just provide text (using SingleTextProvider).
            DEFAULT: we consider an application of stemmer for Russian Language.
            DICT: in which for every type there is an assigned Vectorizer
                vectorization of term types.
                {
                    TermType.Word: Vectorizer,
                    TermType.Entity: Vectorizer,
                    ...
                }
    """
    assert(isinstance(str_entity_fmt, StringEntitiesFormatter))
    assert(isinstance(ctx, NetworkSerializationContext))
    assert(isinstance(vectorizers, dict) or vectorizers == "default" or vectorizers is None)

    term_embedding_pairs = None

    if vectorizers is not None:

        if vectorizers == "default":
            # initialize default vectorizer for Russian language.
            embedding = load_embedding_news_mystem_skipgram_1000_20_2015(stemmer=MystemWrapper(), auto_download=True)
            bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
            norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                                     token_offset=12345)
            vectorizers = {
                TermTypes.WORD: bpe_vectorizer,
                TermTypes.ENTITY: bpe_vectorizer,
                TermTypes.FRAME: bpe_vectorizer,
                TermTypes.TOKEN: norm_vectorizer
            }

        # Setup term-embedding pairs collection instance.
        term_embedding_pairs = collections.OrderedDict()

        # Use text provider with vectorizers.
        text_provider = NetworkSingleTextProvider(
            text_terms_mapper=VectorizedNetworkTermMapping(
                vectorizers=vectorizers,
                string_entities_formatter=str_entity_fmt),
            pair_handling_func=lambda pair: __add_term_embedding(
                dict_data=term_embedding_pairs,
                term=pair[0],
                emb_vector=pair[1]))
    else:
        # Create text provider which without vectorizers.
        text_provider = BaseSingleTextProvider(
            text_terms_mapper=OpinionContainingTextTermsMapper(str_entity_fmt))

    return NetworkSampleRowProvider(
        label_provider=ctx.LabelProvider,
        text_provider=text_provider,
        frames_connotation_provider=ctx.FramesConnotationProvider,
        frame_role_label_scaler=ctx.FrameRolesLabelScaler,
        pos_terms_mapper=PosTermsMapper(ctx.PosTagger) if ctx.PosTagger is not None else None,
        term_embedding_pairs=term_embedding_pairs)
