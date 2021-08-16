from . import utils
from os.path import join


def download():
    root_dir = utils.get_default_download_dir()

    data = {
        # RuSentiLex
        "rusentilex.zip": "https://www.dropbox.com/s/bdsl3kney30y45z/rusentilex.zip?dl=0",
        # RuSentRel-v1.1
        "rusentrel-v1_1.zip": "https://www.dropbox.com/s/6aw5jv84jf5hrl2/rusentrel-v1_1.zip?dl=1",
        # RuSentiFrames
        "rusentiframes-v1_0.zip": "https://www.dropbox.com/s/zvkis77li3f40bm/rusentiframes-v1_0.zip?dl=1",
        "rusentiframes-v2_0.zip": "https://www.dropbox.com/s/slbyma7eudmmugp/rusentiframes-v2_0.zip?dl=1",
        # RuAttitudes-v1.0 (Many variations)
        "ruattitudes-dbg.zip": "https://www.dropbox.com/s/5lmqw9kyb4tfm94/ruattitudes-dbg.zip?dl=1",
        "ruattitudes-v1_0.zip": "https://www.dropbox.com/s/wg6oa447msdytj3/ruattitudes-v1_0.zip?dl=1",
        "ruattitudes-v1_1.zip": "https://www.dropbox.com/s/e3menx5iqyush19/ruattitudes-v1_1.zip?dl=1",
        "ruattitudes-v1_2.zip": "https://www.dropbox.com/s/1psvsvy7n3hmpen/ruattitudes-v1_1-n.zip?dl=1",
        # RuAttitudes-v2.0 Base
        "ruattitudes-v2_0_base.zip": "https://www.dropbox.com/s/y39vqzzjumqhce1/ruattitudes_20_base.zip?dl=1",
        "ruattitudes-v2_0_base_neut.zip": "https://www.dropbox.com/s/3xh7gd004oyuwx5/ruattitudes_20_base_neut.zip?dl=1",
        # RuAttitudes-v2.0 Large
        "ruattitudes-v2_0_large.zip": "https://www.dropbox.com/s/43iqoxlyh38qk8u/ruattitudes_20_large.zip?dl=1",
        "ruattitudes-v2_0_large_neut.zip": "https://www.dropbox.com/s/6edqsxehtus4c61/ruattitudes_20_large_neut.zip?dl=1"
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        utils.download(dest_file_path=join(root_dir, local_name),
                       source_url=url_link)
