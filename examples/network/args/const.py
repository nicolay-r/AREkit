import os
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
NEURAL_NETWORKS_TARGET_DIR = DATA_DIR
