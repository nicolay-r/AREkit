import zipfile
from os import path
from os.path import dirname


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
        assert(isinstance(inner_path, unicode))
        assert(callable(process_func))
        assert(isinstance(version, unicode))

        with zipfile.ZipFile(cls.get_archive_filepath(version), "r") as zip_ref:
            with zip_ref.open(inner_path, mode='r') as c_file:
                return process_func(c_file)

    @classmethod
    def iter_from_zip(cls, inner_path, process_func, version):
        assert(isinstance(inner_path, unicode))
        assert(callable(process_func))
        assert(isinstance(version, unicode))

        with zipfile.ZipFile(cls.get_archive_filepath(version), "r") as zip_ref:
            with zip_ref.open(inner_path, mode='r') as c_file:
                for result in process_func(c_file):
                    yield result

    @staticmethod
    def get_data_root():
        return path.join(dirname(__file__), u"../data/")
