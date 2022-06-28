import zipfile

import enum

from arekit.common import utils


class ZipArchiveUtils(object):

    @staticmethod
    def get_archive_filepath(version):
        raise NotImplementedError()

    @classmethod
    def read_from_zip(cls, inner_path, process_func, version):
        """
        process_func:
            func which receives a file reader
        """
        assert(isinstance(inner_path, str))
        assert(callable(process_func))
        assert(isinstance(version, enum.Enum))

        with zipfile.ZipFile(cls.get_archive_filepath(version.value), "r") as zip_ref:
            with zip_ref.open(inner_path, mode='r') as c_file:
                return process_func(c_file)

    @classmethod
    def iter_from_zip(cls, inner_path, process_func, version):
        assert(isinstance(inner_path, str))
        assert(callable(process_func))
        assert(isinstance(version, enum.Enum))

        with zipfile.ZipFile(cls.get_archive_filepath(version.value), "r") as zip_ref:
            with zip_ref.open(inner_path, mode='r') as c_file:
                for result in process_func(c_file):
                    yield result

    @classmethod
    def iter_filenames_from_zip(cls, version):
        assert(isinstance(version, enum.Enum))
        with zipfile.ZipFile(cls.get_archive_filepath(version.value), "r") as zip_ref:
            return iter(zip_ref.namelist())

    @staticmethod
    def get_data_root():
        return utils.get_default_download_dir()
