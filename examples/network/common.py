from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter
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
