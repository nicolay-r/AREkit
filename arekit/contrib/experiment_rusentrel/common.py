from arekit.common.entities.base import Entity
from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.pipeline_item import TextParserPipelineItem
from arekit.common.text.stemmer import Stemmer
from arekit.processing.text.pipeline_frames import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer


def entity_to_group_func(entity, synonyms):
    assert(isinstance(entity, Entity) or entity is None)
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)

    if entity is None:
        return None

    # By default, we provide the related group index.
    group_index = entity.GroupIndex
    if group_index is not None:
        return group_index

    if synonyms is None:
        return None

    # Otherwise, we search for the related group index
    # using synonyms collection.
    value = entity.Value
    if not synonyms.contains_synonym_value(value):
        return None
    return synonyms.get_synonym_group_index(value)


def create_text_parser(exp_data, entities_parser):
    assert(isinstance(entities_parser, TextParserPipelineItem))

    if not isinstance(exp_data, SerializationData):
        # We do not utlize text_parser in such case.
        return None

    pipeline = [entities_parser,
                DefaultTextTokenizer(keep_tokens=True),
                LemmasBasedFrameVariantsParser(frame_variants=exp_data.FrameVariantCollection,
                                               stemmer=exp_data.Stemmer)]

    return BaseTextParser(pipeline)
