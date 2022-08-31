import unittest
from collections import OrderedDict

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import Label, NoLabel
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.entities.formatters.str_simple_uppercase_fmt import SimpleUppercasedEntityFormatter
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer
from tests.tutorials.test_tutorial_pipeline_text_opinion_annotation import FooDocumentOperations, CustomLabelsFormatter


class Positive(Label):
    pass


class Negative(Label):
    pass


class CustomSentimentLabelScaler(SentimentLabelScaler):
    def __init__(self):
        int_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), -1)])
        uint_to_label = OrderedDict([(NoLabel(), 0), (Positive(), 1), (Negative(), 2)])
        super(SentimentLabelScaler, self).__init__(int_dict=int_to_label, uint_dict=uint_to_label)

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)


class TestSamplingNetwork(unittest.TestCase):

    def test(self):

        stemmer = MystemWrapper()
        embedding = load_embedding_news_mystem_skipgram_1000_20_2015(stemmer)

        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=Positive, neg_label_type=Negative),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=Positive, neg_label_type=Negative))

        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        ctx = NetworkSerializationContext(
            labels_scaler=CustomSentimentLabelScaler(),
            pos_tagger=POSMystemWrapper(mystem=stemmer.MystemInstance),
            frame_roles_label_scaler=CustomSentimentLabelScaler(),
            frames_connotation_provider=RuSentiFramesConnotationProvider(frames_collection))

        writer = TsvWriter(write_header=True)
        samples_io = SamplesIO("out/", writer, target_extension=".tsv.gz")

        embedding_io = NpEmbeddingIO(target_dir="out/")

        bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
        norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                                 token_offset=12345)
        vectorizers = {
            TermTypes.WORD: bpe_vectorizer,
            TermTypes.ENTITY: bpe_vectorizer,
            TermTypes.FRAME: bpe_vectorizer,
            TermTypes.TOKEN: norm_vectorizer
        }

        pipeline_item = NetworksInputSerializerPipelineItem(
            vectorizers=vectorizers,
            samples_io=samples_io,
            emb_io=embedding_io,
            ctx=ctx,
            str_entity_fmt=SimpleUppercasedEntityFormatter(),
            balance_func=lambda data_type: data_type == DataType.Train,
            save_labels_func=lambda data_type: data_type != DataType.Test,
            save_embedding=True)

        pipeline = BasePipeline([
            pipeline_item
        ])

        #####
        # Declaring pipeline related context parameters.
        #####
        no_folding = NoFolding(doc_ids=[0, 1], supported_data_type=DataType.Train)
        doc_ops = FooDocumentOperations()
        text_parser = BaseTextParser(pipeline=[
            BratTextEntitiesParser(),
            DefaultTextTokenizer(keep_tokens=True),
            LemmasBasedFrameVariantsParser(frame_variants=frame_variant_collection, stemmer=stemmer)])
        train_pipeline = text_opinion_extraction_pipeline(
            annotators=[
                PredefinedTextOpinionAnnotator(
                    doc_ops,
                    label_formatter=CustomLabelsFormatter(pos_label_type=Positive, neg_label_type=Negative))
            ],
            text_opinion_filters=[
                DistanceLimitedTextOpinionFilter(terms_per_context=50)
            ],
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser)
        #####

        pipeline.run(input_data=None,
                     params_dict={
                         "data_folding": no_folding,
                         "data_type_pipelines": {DataType.Train: train_pipeline}
                     })
