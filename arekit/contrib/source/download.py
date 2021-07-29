import utils
from os.path import join


def download():
    root_dir = utils.get_default_download_dir()

    data = {
        # v1.0
        "ruattitudes-v1_2.zip": "https://github.com/nicolay-r/ner-bilstm-crf-tensorflow/blob/master/ner/utils.py",
        # Base
        "ruattitudes-v2_0_base.zip": "https://www.dropbox.com/s/y39vqzzjumqhce1/ruattitudes_20_base.zip?dl=1",
        "ruattitudes-v2_0_base_neut.zip": "https://www.dropbox.com/s/3xh7gd004oyuwx5/ruattitudes_20_base_neut.zip?dl=1",  # Large
        "ruattitudes-v2_0_large.zip": "https://www.dropbox.com/s/43iqoxlyh38qk8u/ruattitudes_20_large.zip?dl=1",
        "ruattitudes-v2_0_large_neut.zip": "https://www.dropbox.com/s/6edqsxehtus4c61/ruattitudes_20_large_neut.zip?dl=1"
    }

    # Perform downloading ...
    for local_name, url_link in data.iteritems():
        utils.download(dest_file_path=join(root_dir, local_name),
                       source_url=url_link)
