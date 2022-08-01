from os.path import join

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.io_utils.utils import filename_template


class OpinionsIOUtils(object):

    def __init__(self, target_dir, target_extension=".tsv.gz"):
        self.__target_dir = target_dir
        self.__target_extension = target_extension

    def create_view(self, target):
        storage = BaseRowsStorage.from_tsv(filepath=target)
        return BaseOpinionStorageView(storage)

    def create_writer_target(self, data_type, data_folding):
        return self.__get_input_opinions_target(data_type, data_folding=data_folding)

    def __get_input_opinions_target(self, data_type, data_folding):
        template = filename_template(data_type=data_type, data_folding=data_folding)
        return self.__get_filepath(out_dir=self.__target_dir,
                                   template=template,
                                   prefix="opinion",
                                   extension=self.__target_extension)

    @staticmethod
    def __get_filepath(out_dir, template, prefix, extension):
        assert(isinstance(template, str))
        assert(isinstance(prefix, str))
        assert(isinstance(extension, str))
        return join(out_dir, "{prefix}-{template}{extension}".format(
            prefix=prefix, template=template, extension=extension))
