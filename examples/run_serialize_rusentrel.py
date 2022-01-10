import argparse

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.folding.types import FoldingType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.single_label import PairSingleLabelProvider
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.factory import create_experiment
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel
from arekit.contrib.networks.run_serializer import NetworksExperimentInputSerializer
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.network.args.common import ExperimentTypeArg, LabelsCountArg, RusVectoresEmbeddingFilepathArg, \
    TermsPerContextArg, RuSentiFramesVersionArg, StemmerArg, UseBalancingArg, DistanceInTermsBetweenAttitudeEndsArg
from examples.network.train.common import Common
from examples.rusentrel.data import RuSentRelExperimentSerializationData
from examples.rusentrel.exp_io import CustomRuSentRelNetworkExperimentIO


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="RuSentRel dataset serialization script")

    # Default parameters
    rusentrel_version = RuSentRelVersions.V11
    ra_version = RuAttitudesVersions.V20LargeNeut

    # Provide arguments.
    ExperimentTypeArg.add_argument(parser)
    LabelsCountArg.add_argument(parser)
    RusVectoresEmbeddingFilepathArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    RuSentiFramesVersionArg.add_argument(parser)
    EntityFormatterTypesArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    UseBalancingArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    embedding_filepath = RusVectoresEmbeddingFilepathArg.read_argument(args)
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    frames_version = RuSentiFramesVersionArg.read_argument(args)
    entity_fmt = EntityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    use_balancing = UseBalancingArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    pos_tagger = POSMystemWrapper(MystemWrapper().MystemInstance)

    annot_algo = PairBasedAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_provider=PairSingleLabelProvider(label_instance=ExperimentNeutralLabel()))

    # Preparing necessary structures for further initializations.
    experiment_data = RuSentRelExperimentSerializationData(
        labels_scaler=Common.create_labels_scaler(labels_count),
        embedding=Common.load_rusvectores_embedding(filepath=embedding_filepath, stemmer=stemmer),
        terms_per_context=terms_per_context,
        frames_version=frames_version,
        rusentrel_version=rusentrel_version,
        str_entity_formatter=create_entity_formatter(entity_fmt),
        stemmer=stemmer,
        pos_tagger=pos_tagger,
        annotator=DefaultAnnotator(annot_algo=annot_algo))

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=use_balancing,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    experiment = create_experiment(
        exp_type=exp_type,
        experiment_data=experiment_data,
        folding_type=FoldingType.Fixed,
        rusentrel_version=RuSentRelVersions.V11,
        ruattitudes_version=ra_version,
        experiment_io_type=CustomRuSentRelNetworkExperimentIO,
        extra_name_suffix=extra_name_suffix,
        load_ruattitude_docs=True)

    # Performing serialization process.
    serialization_engine = NetworksExperimentInputSerializer(
        experiment=experiment,
        balance=use_balancing,
        force_serialize=True,
        skip_folder_if_exists=True)

    serialization_engine.run()
