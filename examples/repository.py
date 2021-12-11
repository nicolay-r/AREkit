from arekit.common.data.input.providers.label.base import LabelProvider
from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.entities.collection import EntityCollection
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News
from arekit.common.text.options import TextParseOptions
from arekit.contrib.experiment_rusentrel.annot.three_scale import ThreeScaleTaskAnnotator
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from examples.input import EXAMPLES
from examples.network.utils import SingleDocOperations, CustomOpinionOperations, CustomSerializationData, \
    CustomExperiment, CustomTextParser


def pipeline_serialize(text, label_provider):
    assert(isinstance(label_provider, LabelProvider))

    # Step 1. Parse text.
    sentences = text  # TODO. split text onto sentences.
    stemmer = MystemWrapper()

    news = News(doc_id=0, sentences=sentences)

    parse_options = TextParseOptions(
        parse_entities=False,
        frame_variants_collection=FrameVariantsCollection(),
        stemmer=stemmer)

    text_parser = CustomTextParser(parse_options)

    parsed_news = text_parser.parse_news(news=news)

    # Step 2. Annotate text.
    synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
        stemmer=stemmer,
        version=RuSentRelVersions.V11)

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    parsed_news.get_entity_position()

    entities = EntityCollection(entities=None,
                                synonyms=synonyms)

    opins_for_extraction = annot_algo.iter_opinions(
        parsed_news=parsed_news,
        entities_collection=entities)   # TODO. Create custom entity collections.

    doc_ops = SingleDocOperations(news=news, text_parser=text_parser)

    opin_ops = CustomOpinionOperations(labels_formatter=None,
                                       iter_opins=opins_for_extraction,
                                       synonyms=synonyms)

    exp_data = CustomSerializationData(label_scaler=label_provider.LabelScaler,
                                       stemmer=stemmer,
                                       annot=ThreeScaleTaskAnnotator(annot_algo=annot_algo))

    # Step 3. Serialize data
    experiment = CustomExperiment(synonyms=synonyms,
                                  exp_data=exp_data,
                                  exp_io_type=NetworkIOUtils,
                                  doc_ops=doc_ops,
                                  opin_ops=opin_ops)

    NetworkInputHelper.prepare(experiment=experiment,
                               terms_per_context=50,
                               balance=None)


if __name__ == '__main__':

    text= EXAMPLES["simple"]

    labels_scaler = ThreeLabelScaler()
    label_provider = MultipleLabelProvider(label_scaler=labels_scaler)

    pipeline_serialize(text=text, label_provider=label_provider)
