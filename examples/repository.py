from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.base import News
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.experiment_rusentrel.annot.three_scale import ThreeScaleTaskAnnotator
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer
from examples.input import EXAMPLES
from examples.network.utils import SingleDocOperations, CustomSerializationData, \
    CustomExperiment, TextEntitiesParser, TermsSplitterParser


def create_frame_variants_collection():

    frames = RuSentiFramesCollection.read_collection(RuSentiFramesVersions.V20)
    frame_variant_collection = FrameVariantsCollection()
    frame_variant_collection.fill_from_iterable(
        variants_with_id=frames.iter_frame_id_and_variants(),
        overwrite_existed_variant=True,
        raise_error_on_existed_variant=False)

    return frame_variant_collection


def pipeline_serialize(sentences_text_list, label_provider):
    assert(isinstance(sentences_text_list, list))
    assert(isinstance(label_provider, LabelProvider))

    # TODO. split text onto sentences.
    sentences = list(map(lambda text: BaseNewsSentence(text), sentences_text_list))

    stemmer = MystemWrapper()

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    frame_variants_collection = create_frame_variants_collection()

    # Step 2. Annotate text.
    synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
        stemmer=stemmer,
        version=RuSentRelVersions.V11)

    # Step 1. Parse text.
    news = News(doc_id=0, sentences=sentences)

    text_parser = BaseTextParser(
        pipeline=[
            TermsSplitterParser(),
            TextEntitiesParser(),
            EntitiesGroupingPipelineItem(synonyms.get_synonym_group_index),
            DefaultTextTokenizer(keep_tokens=True),
            LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=stemmer,
                                           frame_variants=frame_variants_collection)
        ])

    exp_data = CustomSerializationData(label_scaler=label_provider.LabelScaler,
                                       stemmer=stemmer,
                                       annot=ThreeScaleTaskAnnotator(annot_algo=annot_algo),
                                       frame_variants_collection=frame_variants_collection)

    labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

    # Step 3. Serialize data
    experiment = CustomExperiment(
        exp_data=exp_data,
        doc_ops=SingleDocOperations(news=news, text_parser=text_parser),
        labels_formatter=labels_fmt,
        synonyms=synonyms,
        neutral_labels_fmt=labels_fmt)

    NetworkInputHelper.prepare(experiment=experiment,
                               terms_per_context=50,
                               balance=False,
                               value_to_group_id_func=synonyms.get_synonym_group_index)


if __name__ == '__main__':

    text = EXAMPLES["simple"]

    labels_scaler = ThreeLabelScaler()
    label_provider = MultipleLabelProvider(label_scaler=labels_scaler)

    pipeline_serialize(sentences_text_list=text, label_provider=label_provider)
