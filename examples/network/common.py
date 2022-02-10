from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions


def create_infer_experiment_name_provider():
    return ExperimentNameProvider(name="example", suffix="infer")


def create_frames_collection():
    return RuSentiFramesCollection.read_collection(
        version=RuSentiFramesVersions.V20,
        labels_fmt=ExperimentRuSentiFramesLabelsFormatter())


def create_and_fill_variant_collection(frames_collection):
    frame_variant_collection = FrameVariantsCollection()
    frame_variant_collection.fill_from_iterable(
        variants_with_id=frames_collection.iter_frame_id_and_variants(),
        overwrite_existed_variant=True,
        raise_error_on_existed_variant=False)
    return frame_variant_collection


def create_bags_collection_type(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.SingleInstance:
        return SingleBagsCollection
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MultiInstanceBagsCollection
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return MultiInstanceBagsCollection


def create_network_model_io(full_model_name, source_dir, embedding_filepath,
                            target_dir, vocab_filepath, model_name_tag):

    return NeuralNetworkModelIO(full_model_name=full_model_name,
                                target_dir=target_dir,
                                source_dir=source_dir,
                                embedding_filepath=embedding_filepath,
                                vocab_filepath=vocab_filepath,
                                model_name_tag=model_name_tag)