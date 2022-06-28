from os.path import join, exists

from arekit.common import utils


NEWS_MYSTEM_SKIPGRAM_1000_20_2015 = "news_mystem_skipgram_1000_20_2015.bin.gz"


def get_resource_path(local_name, check_existance=False):
    assert(isinstance(local_name, str))
    filepath = join(utils.get_default_download_dir(), local_name)

    if check_existance and not exists(filepath):
        raise Exception("Resource could not be found: {}".format(filepath))

    return filepath


def download():

    data = {
        # Embedding.
        NEWS_MYSTEM_SKIPGRAM_1000_20_2015:
            "http://rusvectores.org/static/models/rusvectores2/{}".format(NEWS_MYSTEM_SKIPGRAM_1000_20_2015),
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        utils.download(dest_file_path=get_resource_path(local_name),
                       source_url=url_link)
