from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.base import News
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames import FrameVariantsParser
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_frames_negation import FrameVariantsSentimentNegation
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer
from examples.exp.doc_ops import SingleDocOperations
from examples.exp.exp import CustomExperiment
from examples.exp.exp_io import InferIOUtils
from examples.network.common import create_frames_collection, create_and_fill_variant_collection
from examples.network.embedding import RusvectoresEmbedding
from examples.network.serialization_data import CustomSerializationContext


class TextSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser, synonyms, opin_annot, name_provider,
                 embedding_path, entity_fmt, stemmer, data_folding):
        assert(isinstance(entities_parser, BasePipelineItem))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(embedding_path, str))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(name_provider, ExperimentNameProvider))

        # Initalize embedding.
        embedding = RusvectoresEmbedding.from_word2vec_format(filepath=embedding_path, binary=True)
        embedding.set_stemmer(stemmer)

        # Initialize synonyms collection.
        self.__synonyms = synonyms
        pos_tagger = POSMystemWrapper(MystemWrapper().MystemInstance)

        # Label provider setup.
        self.__labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

        # Initialize text parser with the related dependencies.
        frames_collection = create_frames_collection()
        frame_variants_collection = create_and_fill_variant_collection(frames_collection)
        self.__text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            entities_parser,
            EntitiesGroupingPipelineItem(self.__synonyms.get_synonym_group_index),
            DefaultTextTokenizer(keep_tokens=True),
            FrameVariantsParser(frame_variants=frame_variants_collection),
            LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=stemmer,
                                           frame_variants=frame_variants_collection),
            FrameVariantsSentimentNegation()])

        # initialize expriment related data.
        self.__exp_ctx = CustomSerializationContext(
            labels_scaler=SingleLabelScaler(NoLabel()),
            embedding=embedding,
            annotator=opin_annot,
            terms_per_context=terms_per_context,
            str_entity_formatter=entity_fmt,
            pos_tagger=pos_tagger,
            name_provider=name_provider,
            data_folding=data_folding)
        self.__exp_io = InferIOUtils(self.__exp_ctx)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list) or isinstance(input_data, str))

        # setup input data.
        sentences = input_data
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(map(lambda text: BaseNewsSentence(text), sentences))

        # Parse text.
        news = News(doc_id=0, sentences=sentences)

        # Step 3. Serialize data
        experiment = CustomExperiment(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=SingleDocOperations(exp_ctx=self.__exp_ctx, news=news, text_parser=self.__text_parser),
            labels_formatter=self.__labels_fmt,
            synonyms=self.__synonyms,
            neutral_labels_fmt=self.__labels_fmt)

        NetworkInputHelper.prepare(exp_ctx=experiment.ExperimentContext,
                                   exp_io=experiment.ExperimentIO,
                                   doc_ops=experiment.DocumentOperations,
                                   opin_ops=experiment.OpinionOperations,
                                   terms_per_context=self.__exp_ctx.TermsPerContext,
                                   balance=False,
                                   value_to_group_id_func=self.__synonyms.get_synonym_group_index)

        return experiment.ExperimentIO
