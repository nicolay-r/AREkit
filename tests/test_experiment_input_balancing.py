import logging

from arekit.bert.input.providers.label.binary import BinaryLabelProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.formatters.helper.balancing import SampleRowBalancerHelper
from arekit.common.experiment.input.formatters.sample import BaseSampleFormatter
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.terms_mapper import StringTextTermsMapper
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.contrib.networks.entities.str_emb_fmt import StringWordEmbeddingEntityFormatter
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection


# Setup logging format
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

stemmer = MystemWrapper()
label_provider = BinaryLabelProvider(label_scaler=ThreeLabelScaler())
synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=stemmer, is_read_only=True)
terms_mapper = StringTextTermsMapper(entity_formatter=StringWordEmbeddingEntityFormatter(), synonyms=synonyms)

formatter = BaseSampleFormatter(
    data_type=DataType.Train,
    label_provider=label_provider,
    text_provider=BaseSingleTextProvider(terms_mapper))

df = formatter._df

df = df.append({"row_id": 1, "id": 1, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
df = df.append({"row_id": 1, "id": 2, "label": 1, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
df = df.append({"row_id": 1, "id": 5, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
df = df.append({"row_id": 1, "id": 6, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)

balanced_df = SampleRowBalancerHelper.balance_oversampling(
    df=df,
    create_blank_df=lambda size: formatter._create_blank_df(size),
    label_provider=label_provider)

print balanced_df


