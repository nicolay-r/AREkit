import sys
import os
import requests
from tqdm import tqdm


def create_dir_if_not_exists(filepath):
    dir = os.path.dirname(filepath)

    # Check whether string is empty.
    if not dir:
        return

    if not os.path.exists(dir):
        os.makedirs(dir)


def filter_whitespaces(terms):
    return [term.strip() for term in terms if term.strip()]


def split_by_whitespaces(text):
    """
    Assumes to perform a word separation including a variety of space entries.
    In terms of the latter we consider any whitespace separator.
    """
    assert(isinstance(text, str))
    return text.split()


def progress_bar_defined(iterable, total, desc="", unit="it"):
    return tqdm(iterable=iterable,
                total=total,
                desc=desc,
                ncols=120,
                position=0,
                leave=True,
                unit=unit,
                miniters=total / 200)


def progress_bar_iter(iterable, desc="", unit='it'):
    return tqdm(iterable=iterable,
                desc=desc,
                position=0,
                leave=True,
                ncols=120,
                unit=unit)


def get_default_download_dir():
    """ Refered to NLTK toolkit approach
        https://github.com/nltk/nltk/blob/8e771679cee1b4a9540633cc3ea17f4421ffd6c0/nltk/downloader.py#L1051
    """

    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    return os.path.join(homedir, ".arekit")


def download(dest_file_path, source_url):
    """ Refered to https://github.com/nicolay-r/ner-bilstm-crf-tensorflow/blob/master/ner/utils.py
        Simple http file downloader
    """
    print(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
