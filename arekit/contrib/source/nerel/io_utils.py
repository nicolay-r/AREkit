import enum
from os import path
from os.path import basename

from arekit.common.experiment.data_type import DataType
from arekit.contrib.source.nerel.folding.fixed import create_fixed_folding
from arekit.contrib.source.zip_utils import ZipArchiveUtils


class NerelVersions(enum.Enum):
    """ List of the supported version of this collection
    """

    V1 = "v1_0"
    V11 = "v1_1"


DEFAULT_VERSION = NerelVersions.V1


class NerelIOUtils(ZipArchiveUtils):

    splits = {
        DataType.Train: "train",
        DataType.Dev: "dev",
        DataType.Test: "test"
    }

    @staticmethod
    def get_archive_filepath(version):
        return path.join(NerelIOUtils.get_data_root(), "nerel-{}.zip".format(version))

    @staticmethod
    def get_annotation_innerpath(folding_data_type, filename):
        assert(isinstance(filename, str))
        return path.join(NerelIOUtils.splits[folding_data_type], "{}.ann".format(filename))

    @staticmethod
    def get_news_innerpath(folding_data_type, filename):
        assert(isinstance(filename, str))
        return path.join(NerelIOUtils.splits[folding_data_type], "{}.txt".format(filename))

    @staticmethod
    def __iter_filenames_from_dataset(version):
        assert(isinstance(version, enum.Enum))

        for filename in NerelIOUtils.iter_filenames_from_zip(version):

            extension = filename[-4:]

            # Crop extension.
            filename = filename[:-4]

            if extension != ".txt":
                continue

            yield filename, basename(filename)

    @staticmethod
    def __iter_filename_and_splittype(version):
        filenames_it = NerelIOUtils.__iter_filenames_from_dataset(version=version)
        for doc_id, data in enumerate(filenames_it):
            filepath, filename = data
            for split_type, split_name in NerelIOUtils.splits.items():
                if split_name in filepath:
                    yield filename, split_type

    @staticmethod
    def iter_collection_filenames(version=DEFAULT_VERSION):
        filenames_it = NerelIOUtils.__iter_filenames_from_dataset(version=version)
        for doc_id, filename in enumerate(filenames_it):
            yield doc_id, filename

    @staticmethod
    def map_doc_to_fold_type(version=DEFAULT_VERSION):
        d2f = {}
        for filename, split_type in NerelIOUtils.__iter_filename_and_splittype(version):
            d2f[filename] = split_type
        return d2f

    @staticmethod
    def read_dataset_split(version=DEFAULT_VERSION):
        f2d = {}
        for filename, split_type in NerelIOUtils.__iter_filename_and_splittype(version):
            if split_type not in f2d:
                f2d[split_type] = []
            f2d[split_type].append(filename)

        filenames_by_ids, data_folding = create_fixed_folding(train_filenames=f2d[DataType.Train],
                                                              test_filenames=f2d[DataType.Test],
                                                              dev_filenames=f2d[DataType.Dev])

        return filenames_by_ids, data_folding