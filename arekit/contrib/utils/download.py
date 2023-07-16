import os
import tarfile
from os.path import join, exists

from arekit.common import utils
from arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper
from arekit.contrib.utils.np_utils.vocab import VocabRepositoryUtils

NEWS_MYSTEM_SKIPGRAM_1000_20_2015 = "news_mystem_skipgram_1000_20_2015.tar.gz"


def __get_resource(local_name, check_existance=False, download_if_missed=False):
    assert (isinstance(local_name, str))
    filepath = join(utils.get_default_download_dir(), local_name)

    if check_existance and not exists(filepath):
        if download_if_missed:
            download()
            # We try to ger the resource again but won't attempt to download it again.
            __get_resource(local_name, check_existance=check_existance, download_if_missed=False)
        else:
            raise Exception("Resource could not be found: {}".format(filepath))

    return filepath


def __get_embedding_dir(filepath):
    return filepath.replace(".tar.gz", "")


def load_embedding_and_vocab(local_name, check_existance=False, download_if_missed=False):
    tar_gz_archive = __get_resource(local_name, check_existance=check_existance,
                                    download_if_missed=download_if_missed)
    target_dir = __get_embedding_dir(tar_gz_archive)
    embedding = NpzEmbeddingHelper.load_embedding(os.path.join(target_dir, "embedding.npz"))
    vocab = VocabRepositoryUtils.load(os.path.join(target_dir, "vocab.txt"))
    return embedding, vocab


def download():
    data = {
        NEWS_MYSTEM_SKIPGRAM_1000_20_2015: "https://www.dropbox.com/s/0omnlgzgnjhxlmf/{filename}?dl=1".format(
            filename=NEWS_MYSTEM_SKIPGRAM_1000_20_2015),
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        utils.download(dest_file_path=__get_resource(local_name),
                       source_url=url_link)

    # Untar files ...
    for local_name in data.keys():

        if ".tar.gz" not in local_name:
            continue

        target_filepath = __get_resource(local_name)
        with tarfile.open(target_filepath) as file:
            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(file, __get_embedding_dir(target_filepath))
