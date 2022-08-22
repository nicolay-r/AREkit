from enum import Enum
from os import path
from os.path import basename

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class SentiNerelVersions(Enum):
    """ List of the supported version of this collection
    """

    V1 = "v1_0"


class SentiNerelIOUtils(ZipArchiveUtils):

    inner_root = "sentiment_dataset"

    @staticmethod
    def get_archive_filepath(version):
        return path.join(SentiNerelIOUtils.get_data_root(), "sentinerel-{}.zip".format(version))

    @staticmethod
    def get_annotation_innerpath(filename):
        assert(isinstance(filename, str))
        return path.join(SentiNerelIOUtils.inner_root, "{}.ann".format(filename))

    @staticmethod
    def get_news_innerpath(filename):
        assert(isinstance(filename, str))
        return path.join(SentiNerelIOUtils.inner_root, "{}.txt".format(filename))

    @staticmethod
    def __iter_filenames_from_dataset(folder_name):
        assert(isinstance(folder_name, str))

        for filename in SentiNerelIOUtils.iter_filenames_from_zip(SentiNerelVersions.V1):

            extension = filename[-4:]

            # Crop extension.
            filename = filename[:-4]

            if extension != ".txt":
                continue

            if not folder_name in filename:
                continue

            yield basename(filename)

    # region public methods

    @staticmethod
    def iter_collection_filenames():
        filenames_it = SentiNerelIOUtils.__iter_filenames_from_dataset(
            folder_name=SentiNerelIOUtils.inner_root)

        for doc_id, filename in enumerate(filenames_it):
            yield doc_id, filename

    # endregion
