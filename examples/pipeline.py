from arekit.common.entities.base import Entity
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.columns.opinion import OpinionColumnsProvider
from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.opinions import BaseOpinionsRowProvider
from arekit.common.experiment.input.repositories.opinions import BaseInputOpinionsRepository
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.experiment.input.storages.tsv_opinion import TsvOpinionsStorage
from arekit.common.experiment.input.storages.tsv_sample import TsvSampleStorage
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.synonyms import SynonymsCollection

from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.input.formatters.pos_mapper import PosTermsMapper
from arekit.contrib.networks.core.input.providers.sample import NetworkSampleRowProvider
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.predict.tsv_provider import TsvPredictProvider

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

    ########################
    # Step 1. Parse text.
    ########################

    sentences = text  # TODO. split text onto sentences.

    news = News(news_id=0,
                sentences=sentences)

    parse_options = NewsParseOptions(
        parse_entities=False,
        frame_variants_collection=FrameVariantsCollection(),
        stemmer=Stemmer())

    parsed_news = TextParser.parse_news(news=news,
                                        parse_options=parse_options)

    ########################
    # Step 2. Annotate text.
    ########################

    labels_scaler = ThreeLabelScaler()

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    opins_for_extraction = annot_algo.iter_opinions(
        parsed_news=parsed_news,
        entities_collection=None)   # TODO. Create custom entity collections.

    doc_ops = None
    opin_ops = None

    # We pass it into NetworkInputEncoder
    opinion_provider = OpinionProvider.from_experiment(    # Refactor API.
        doc_ops=doc_ops,             # TODO: Remove doc_ops
        opin_ops=opin_ops,           # TODO: Remove doc_ops
        data_type=DataType.Test,
        parsed_news_it_func=lambda: [parsed_news],
        terms_per_context=50)

    sample_row_provider = NetworkSampleRowProvider(
        label_provider=MultipleLabelProvider(label_scaler=labels_scaler),
        text_provider=None,
        frames_collection=None,
        frame_role_label_scaler=None,
        entity_to_group_func=entity_to_group_func,
        pos_terms_mapper=PosTermsMapper(None))
    opinion_row_provider = BaseOpinionsRowProvider()

    samples_repo = BaseInputSamplesRepository(
        columns_provider=SampleColumnsProvider(store_labels=True),
        rows_provider=sample_row_provider,
        storage=TsvSampleStorage(balance=False, write_header=True))

    opinions_repo = BaseInputOpinionsRepository(
        columns_provider=OpinionColumnsProvider(),
        rows_provider=opinion_row_provider,
        storage=TsvOpinionsStorage())

    # Populate repositories
    samples_repo.populate(opinion_provider=opinion_provider,
                          target="samples.txt",
                          desc="sample")

    opinions_repo.populate(opinion_provider=opinion_provider,
                           target="opinions.txt",
                           desc="opinion")

    ###########################
    # Step 3. Data preparation.
    ###########################

    handled_data = HandledData.create_empty()

    # TODO. Provide samples reader.
    handled_data.perform_reading_and_initialization(
        dtypes=[DataType.Test],                       # TODO. Will be removed.
        exp_io=None,                                  # TODO. Remove
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

    labeling_collection = model.get_samples_labeling_collection(data_type=DataType.Test)

    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    # TODO. For now it is limited to tsv.
    with TsvPredictProvider(filepath="out.txt") as out:
        out.load(sample_id_with_uint_labels_iter=labeling_collection.iter_non_duplicated_labeled_sample_row_ids(),
                 labels_scaler=labels_scaler)


if __name__ == '__main__':

    extract("сша намерена ввести санкции против роccии")
