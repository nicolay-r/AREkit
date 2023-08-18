from arekit.contrib.source.nerel.reader import NerelDocReader
from arekit.contrib.source.nerelbio.io_utils import NerelBioIOUtils


class NerelBioDocReader(NerelDocReader):

    def __init__(self, version):
        super(NerelBioDocReader, self).__init__(version=version, io_utils=NerelBioIOUtils())
