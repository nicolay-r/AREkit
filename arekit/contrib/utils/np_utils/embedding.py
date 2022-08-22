import logging

from arekit.contrib.utils.np_utils.npz_utils import NpzRepositoryUtils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NpzEmbeddingHelper:

    @staticmethod
    def save_embedding(data, target):
        NpzRepositoryUtils.save(data=data, target=target)
        logger.info("Saving embedding [size={shape}]: {filepath}".format(shape=data.shape,
                                                                         filepath=target))

    @staticmethod
    def load_embedding(source):
        embedding = NpzRepositoryUtils.load(source)
        logger.info("Embedding read [size={size}]: {filepath}".format(size=embedding.shape,
                                                                      filepath=source))
        return embedding
