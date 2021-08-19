import collections

from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.entities.base import Entity
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.opinions.base import Opinion
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.input.encoder import NetworkInputEncoder
from arekit.contrib.networks.core.input.formatters.sample import NetworkSampleFormatter
from arekit.contrib.networks.core.input.providers.text.single import NetworkSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.output.encoder import NetworkOutputEncoder
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.parser import TextParser


def __add_term_embedding(dict_data, term, emb_vector):
    if term in dict_data:
        return
    dict_data[term] = emb_vector


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


def extract(text):

    # Step 1. Parse text.

    sentences = text  # TODO. split text onto sentences.

    news = News(news_id=0,
                sentences=sentences,
                entities_parser=None)   # TODO. Implement entities parser.

    parse_options = NewsParseOptions(
        parse_entities=False,
        frame_variants_collection=FrameVariantsCollection(),
        stemmer=Stemmer())

    parsed_news = TextParser.parse_news(news=news,
                                        parse_options=parse_options)

    ########################
    # Step 2. Annotate text.
    ########################

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    opinions_list = annot_algo.iter_opinions(
        parsed_news=parsed_news,
        entities_collection=None)   # TDOO. Create custom entity collections.

    doc_ops = None
    opin_ops = None

    # TODO. We need to pass this into OpinionProvder for initilization.
    opins_for_extraction = opinions_list

    # We pass it into NetworkInputEncoder
    opin_provider = OpinionProvider.from_experiment(    # Refactor API.
        doc_ops=doc_ops,             # TODO: Remove doc_ops
        opin_ops=opin_ops,           # TODO: Remove doc_ops
        data_type=DataType.Test,
        parsed_news_it_func=lambda: [parsed_news],
        terms_per_context=50)

    exp_data = None                  # embeddings, frames_collection, vocab, label_scaler, etc.
    NetworkInputEncoder.to_tsv_with_embedding_and_vocabulary(
        exp_io=None,                 # TOdO. Remove from method API.
        opin_ops=None,               # TODO. Remove from method API.
        doc_ops=None,                # TODO. Remove from method API.
        iter_parsed_news_func=None,  # TODO. Remove from method API.
        terms_per_context=None,      # TODO. Remove from method API.
        balance=False,
        exp_data=exp_data,
        data_type=DataType.Test,
        entity_to_group_func=entity_to_group_func,
        term_embedding_pairs=collections.OrderedDict())

    ###########################
    # Step 3. Data preparation.
    ###########################

    handled_data = HandledData.create_empty()

    # TODO. Provide samples reader.
    handled_data.perform_reading_and_initialization(
        doc_ops=None,                                 # TODO. Will be removed.
        exp_io=None,
        vocab=None,
        labels_count=3,
        bags_collection_type=SingleBagsCollection,
        config=None,                                  # TODO. Конфигурация сети.
    )

    ############################
    # Step 4. Model preparation.
    ############################

    model = BaseTensorflowModel(
        nn_io=None,
        network=None,                                 # Нейросеть.
        handled_data=handled_data,
        bags_collection_type=SingleBagsCollection,    # Используем на вход 1 пример.
        config=None,                                  # TODO. Конфигурация сети.
    )

    model.predict()

    ########################################################
    # Step 5. Gather annotated contexts onto document level.
    ########################################################

    labels_scaler = ThreeLabelScaler()
    labeling_collection = model.get_samples_labeling_collection(data_type=DataType.Test)

    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    NetworkOutputEncoder.to_tsv(
        filepath=None,
        sample_id_with_uint_labels_iter=labeling_collection.iter_non_duplicated_labeled_sample_row_ids(),
        labels_scaler=labels_scaler)


if __name__ == '__main__':

    extract("сша намерена ввести санкции против роccии")
