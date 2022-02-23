from collections import OrderedDict

from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.single_label import PairSingleLabelProvider
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.base import News
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames import FrameVariantsParser
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_frames_negation import FrameVariantsSentimentNegation
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer
from examples.network.common import create_frames_collection, create_and_fill_variant_collection, \
    create_infer_experiment_name_provider
from examples.network.embedding import RusvectoresEmbedding
from examples.network.infer.doc_ops import SingleDocOperations
from examples.network.infer.exp import CustomExperiment
from examples.network.infer.exp_io import InferIOUtils
from examples.network.serialization_data import CustomSerializationContext
from examples.text.pipeline_entities_default import TextEntitiesParser


class TextSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser,
                 embedding_path, entity_fmt, stemmer, data_folding):
        assert(isinstance(entities_parser, BasePipelineItem) or entities_parser is None)
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(embedding_path, str))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(data_folding, BaseDataFolding))

        # Initalize embedding.
        self.__embedding = RusvectoresEmbedding.from_word2vec_format(filepath=embedding_path, binary=True)
        self.__embedding.set_stemmer(stemmer)

        # Initialize synonyms collection.
        self.__synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
            stemmer=stemmer, version=RuSentRelVersions.V11)

        self.__entities_parser = TextEntitiesParser() if entities_parser is None else entities_parser
        self.__stemmer = stemmer
        self.__terms_per_context = terms_per_context
        self.__entity_fmt = entity_fmt
        self.__data_folding = data_folding

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list) or isinstance(input_data, str))

        sentences = input_data
        if isinstance(sentences, str):
            sentences = [sentences]

        # Labels scalers and formatters setup.
        labels_scaler = BaseLabelScaler(uint_dict=OrderedDict([(NoLabel(), 0)]),
                                        int_dict=OrderedDict([(NoLabel(), 0)]))
        labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

        label_provider = MultipleLabelProvider(label_scaler=labels_scaler)

        sentences = list(map(lambda text: BaseNewsSentence(text), sentences))

        annot_algo = PairBasedAnnotationAlgorithm(
            dist_in_terms_bound=None,
            label_provider=PairSingleLabelProvider(label_instance=NoLabel()))

        frames_collection = create_frames_collection()
        frame_variants_collection = create_and_fill_variant_collection(frames_collection)

        # Parse text.
        news = News(doc_id=0, sentences=sentences)

        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            self.__entities_parser,
            EntitiesGroupingPipelineItem(self.__synonyms.get_synonym_group_index),
            DefaultTextTokenizer(keep_tokens=True),
            FrameVariantsParser(frame_variants=frame_variants_collection),
            LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=self.__stemmer,
                                           frame_variants=frame_variants_collection),
            FrameVariantsSentimentNegation()])

        exp_ctx = CustomSerializationContext(
            labels_scaler=label_provider.LabelScaler,
            stemmer=self.__stemmer,
            embedding=self.__embedding,
            annotator=DefaultAnnotator(annot_algo=annot_algo),
            terms_per_context=self.__terms_per_context,
            str_entity_formatter=self.__entity_fmt,
            pos_tagger=POSMystemWrapper(MystemWrapper().MystemInstance),
            name_provider=create_infer_experiment_name_provider(),
            data_folding=self.__data_folding)

        exp_io = InferIOUtils(exp_ctx)

        # Step 3. Serialize data
        experiment = CustomExperiment(
            exp_io=exp_io,
            exp_ctx=exp_ctx,
            doc_ops=SingleDocOperations(exp_ctx=exp_ctx, news=news, text_parser=text_parser),
            labels_formatter=labels_fmt,
            synonyms=self.__synonyms,
            neutral_labels_fmt=labels_fmt)

        NetworkInputHelper.prepare(exp_ctx=experiment.ExperimentContext,
                                   exp_io=experiment.ExperimentIO,
                                   doc_ops=experiment.DocumentOperations,
                                   opin_ops=experiment.OpinionOperations,
                                   terms_per_context=self.__terms_per_context,
                                   balance=False,
                                   value_to_group_id_func=self.__synonyms.get_synonym_group_index)

        return experiment.ExperimentIO
