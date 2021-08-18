from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment.annot.single_label import DefaultSingleLabelAnnotationAlgorithm
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.news.base import News
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.opinions.base import Opinion
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.parser import TextParser


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

    # Step 2. Annotate text.

    annot_algo = DefaultSingleLabelAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_instance=NoLabel())

    opinions_list = annot_algo.iter_opinions(
        parsed_news=parsed_news,
        entities_collection=None)   # TDOO. Create custom entity collections.

    for opinion in opinions_list:
        # Document-Level opinions
        assert(isinstance(opinion, Opinion))

        for text_opinion in news.extract_linked_text_opinions(opinion):

            s_index = TextOpinionHelper.extract_entity_position(
                parsed_news=parsed_news,
                text_opinion=text_opinion,
                end_type=EntityEndType.Target,
                position_type=TermPositionTypes.SentenceIndex)

            terms = list(parsed_news.iter_sentence_terms(s_index, return_id=False))

            # TODO. Prepare contexts.

    # Step 3. Data preparation.

    handled_data = HandledData.create_empty()

    handled_data.perform_reading_and_initialization(
        doc_ops=None,                           # TODO. Will be removed.
        exp_io=None,
        vocab=None,
        labels_count=3,
        bags_collection_type=SingleBagsCollection,
        config=None,                            # TODO. Конфигурация сети.
    )

    # Step 4. Model preparation.

    model = BaseTensorflowModel(
        nn_io=None,
        network=None,                   # Нейросеть.
        handled_data=handled_data,
        bags_collection_type=SingleBagsCollection,    # Используем на вход 1 пример.
        config=None,                    # TODO. Конфигурация сети.
    )

    model.predict()

    # Step 5. Gather annotated contexts onto document level.

    # TODO.


if __name__ == '__main__':

    extract("сша намерена ввести санкции против роccии")
