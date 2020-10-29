from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.model.model_io import BaseModelIO


class BertIOUtils(BaseIOUtils):

    def get_target_dir(self):
        """ Provides a main directory for input

            NOTE:
            We consider to save serialized results into model dir,
            rather than experiment dir in a base implementation,
            as model affects on text_b, entities representation, etc.
        """

        model_io = self._experiment.DataIO.ModelIO
        assert(isinstance(model_io, BaseModelIO))

        return model_io.get_model_dir()
