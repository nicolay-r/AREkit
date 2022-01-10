import os
from os.path import join

from arekit.contrib.source import utils

current_dir = os.path.dirname(os.path.realpath(__file__))

# Predefined default parameters.
TERMS_PER_CONTEXT = 50
DROPOUT_KEEP_PROB = 0.5
EPOCHS_COUNT = 150
BAGS_PER_MINIBATCH = 2
LEARNING_RATE = 0.1
BAG_SIZE = 1
DATA_DIR = os.path.join(current_dir, "../../data")
EMBEDDING_FILENAME = "news_mystem_skipgram_1000_20_2015.bin.gz"
EMBEDDING_FILEPATH = os.path.join(utils.get_default_download_dir(), EMBEDDING_FILENAME)
VOCAB_DEFAULT = join(DATA_DIR, "vocab-0.txt.npz")

# Considering to forcely terminate training process in case when
# training accuracy becomes greater than the limit value.
# By default we have no limits
TRAIN_ACC_LIMIT = 1.0

# Specific of the particular experiment, therefore
# we disable such limitation by default.
TRAIN_F1_LIMIT = None