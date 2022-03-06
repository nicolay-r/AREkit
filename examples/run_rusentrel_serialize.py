import argparse

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.types import FoldingType
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.factory import create_experiment
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.networks.handlers.serializer import NetworksInputSerializerExperimentIteration
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from examples.network.args import const
from examples.network.args.serialize import EntityFormatterTypesArg
from examples.network.args.common import ExperimentTypeArg, LabelsCountArg, RusVectoresEmbeddingFilepathArg, \
    TermsPerContextArg, StemmerArg, UseBalancingArg, DistanceInTermsBetweenAttitudeEndsArg
from examples.network.serialization_data import CustomSerializationContext
from examples.rusentrel.common import Common
from examples.rusentrel.exp_io import CustomRuSentRelNetworkExperimentIO

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="RuSentRel dataset serialization script")

    # Provide arguments.
    ExperimentTypeArg.add_argument(parser, default="rsr")
    LabelsCountArg.add_argument(parser, default=3)
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntityFormatterTypesArg.add_argument(parser, default="simple")
    StemmerArg.add_argument(parser, default="mystem")
    UseBalancingArg.add_argument(parser, default=True)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser, default=None)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    embedding_filepath = RusVectoresEmbeddingFilepathArg.read_argument(args)
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    entity_fmt = EntityFormatterTypesArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    use_balancing = UseBalancingArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    pos_tagger = POSMystemWrapper(MystemWrapper().MystemInstance)

    # Default parameters
    rusentrel_version = RuSentRelVersions.V11
    ra_version = None if exp_type == ExperimentTypes.RuSentRel else RuAttitudesVersions.V20LargeNeut
    folding_type = FoldingType.Fixed

    synonyms_collection = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer)

    annot_algo = PairBasedAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_provider=ConstantLabelProvider(label_instance=ExperimentNeutralLabel()))

    exp_name = Common.create_exp_name(rusentrel_version=rusentrel_version,
                                      ra_version=ra_version,
                                      folding_type=folding_type)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=use_balancing,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    data_folding = Common.create_folding(
        rusentrel_version=rusentrel_version,
        ruattitudes_version=ra_version,
        doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    # Preparing necessary structures for further initializations.
    exp_ctx = CustomSerializationContext(
        labels_scaler=Common.create_labels_scaler(labels_count),
        embedding=Common.load_rusvectores_embedding(filepath=embedding_filepath, stemmer=stemmer),
        terms_per_context=terms_per_context,
        str_entity_formatter=create_entity_formatter(entity_fmt),
        pos_tagger=pos_tagger,
        annotator=DefaultAnnotator(annot_algo=annot_algo),
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
        data_folding=data_folding)

    experiment = create_experiment(
        exp_type=exp_type,
        exp_ctx=exp_ctx,
        exp_io=CustomRuSentRelNetworkExperimentIO(exp_ctx),
        folding_type=folding_type,
        rusentrel_version=rusentrel_version,
        ruattitudes_version=ra_version,
        load_ruattitude_docs=True,
        ra_doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    # Performing serialization process.
    serialization_handler = NetworksInputSerializerExperimentIteration(
        exp_ctx=experiment.ExperimentContext,
        doc_ops=experiment.DocumentOperations,
        opin_ops=experiment.OpinionOperations,
        exp_io=experiment.ExperimentIO,
        balance=use_balancing,
        force_serialize=True,
        skip_folder_if_exists=True,
        value_to_group_id_func=synonyms_collection.get_synonym_group_index)

    engine = ExperimentEngine(exp_ctx.DataFolding)

    engine.run(handlers=[serialization_handler])
