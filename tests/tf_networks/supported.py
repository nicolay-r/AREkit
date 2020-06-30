from arekit.contrib.networks.context.architectures.att_ef_bilstm import AttentionEndsAndFramesBiLSTM
from arekit.contrib.networks.context.architectures.att_ef_cnn import AttentionEndsAndFramesCNN
from arekit.contrib.networks.context.architectures.att_ef_pcnn import AttentionEndsAndFramesPCNN
from arekit.contrib.networks.context.architectures.att_ends_cnn import AttentionEndsCNN
from arekit.contrib.networks.context.architectures.att_frames_bilstm import AttentionFramesBiLSTM
from arekit.contrib.networks.context.architectures.att_frames_cnn import AttentionFramesCNN
from arekit.contrib.networks.context.architectures.att_se_bilstm import AttentionSynonymEndsBiLSTM
from arekit.contrib.networks.context.architectures.att_se_cnn import AttentionSynonymEndsCNN
from arekit.contrib.networks.context.architectures.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTM
from arekit.contrib.networks.context.architectures.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNN
from arekit.contrib.networks.context.architectures.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTM
from arekit.contrib.networks.context.architectures.att_self_z_yang_rcnn import AttentionSelfZYangRCNN
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM
from arekit.contrib.networks.context.architectures.ian_ef import IANEndsAndFrames
from arekit.contrib.networks.context.architectures.ian_ends import IANEndsBased
from arekit.contrib.networks.context.architectures.ian_frames import IANFrames
from arekit.contrib.networks.context.architectures.ian_se import IANSynonymEndsBased
from arekit.contrib.networks.context.architectures.att_ends_pcnn import AttentionEndsPCNN
from arekit.contrib.networks.context.architectures.att_frames_pcnn import AttentionFramesPCNN
from arekit.contrib.networks.context.architectures.att_se_pcnn import AttentionSynonymEndsPCNN
from arekit.contrib.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.contrib.networks.context.architectures.rcnn import RCNN
from arekit.contrib.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.contrib.networks.context.configurations.att_ef_bilstm import AttentionEndsAndFramesBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_ef_cnn import AttentionEndsAndFramesCNNConfig
from arekit.contrib.networks.context.configurations.att_ef_pcnn import AttentionEndsAndFramesPCNNConfig
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_ends_cnn import AttentionEndsCNNConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.configurations.att_ends_pcnn import AttentionEndsPCNNConfig
from arekit.contrib.networks.context.configurations.att_frames_bilstm import AttentionFramesBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_frames_cnn import AttentionFramesCNNConfig
from arekit.contrib.networks.context.configurations.att_frames_pcnn import AttentionFramesPCNNConfig
from arekit.contrib.networks.context.configurations.att_self_p_zhou_rcnn import AttentionSelfPZhouRCNNConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_se_bilstm import AttentionSynonymEndsBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_se_cnn import AttentionSynonymEndsCNNConfig
from arekit.contrib.networks.context.configurations.att_se_pcnn import AttentionSynonymEndsPCNNConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_rcnn import AttentionSelfZYangRCNNConfig
from arekit.contrib.networks.context.configurations.ian_ef import IANEndsAndFramesConfig
from arekit.contrib.networks.context.configurations.ian_ends import IANEndsBasedConfig
from arekit.contrib.networks.context.configurations.ian_se import IANSynonymEndsBasedConfig
from arekit.contrib.networks.context.configurations.ian_frames import IANFramesConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.architectures.cnn import VanillaCNN
from arekit.contrib.networks.context.architectures.rnn import RNN


def get_supported():

    return [# Self-attention
            (SelfAttentionBiLSTMConfig(), SelfAttentionBiLSTM()),
            (AttentionSelfPZhouBiLSTMConfig(), AttentionSelfPZhouBiLSTM()),
            (AttentionSelfZYangBiLSTMConfig(), AttentionSelfZYangBiLSTM()),

            # CNN based
            (CNNConfig(), VanillaCNN()),
            (CNNConfig(), PiecewiseCNN()),

            # RNN-based
            (RNNConfig(), RNN()),
            (BiLSTMConfig(), BiLSTM()),

            # RCNN-based models (Recurrent-CNN)
            (RCNNConfig(), RCNN()),
            (AttentionSelfPZhouRCNNConfig(), AttentionSelfPZhouRCNN()),
            (AttentionSelfZYangRCNNConfig(), AttentionSelfZYangRCNN()),

            # IAN (Interactive attention networks)
            (IANFramesConfig(), IANFrames()),
            (IANEndsAndFramesConfig(), IANEndsAndFrames()),
            (IANEndsBasedConfig(), IANEndsBased()),
            (IANSynonymEndsBasedConfig(), IANSynonymEndsBased()),

            # MLP-Attention-based
            (AttentionEndsAndFramesBiLSTMConfig(), AttentionEndsAndFramesBiLSTM()),
            (AttentionFramesBiLSTMConfig(), AttentionFramesBiLSTM()),
            (AttentionSynonymEndsBiLSTMConfig(), AttentionSynonymEndsBiLSTM()),
            (AttentionEndsAndFramesPCNNConfig(), AttentionEndsAndFramesPCNN()),
            [AttentionEndsAndFramesCNNConfig(), AttentionEndsAndFramesCNN()],
            (AttentionEndsCNNConfig(), AttentionEndsCNN()),
            (AttentionEndsPCNNConfig(), AttentionEndsPCNN()),
            (AttentionSynonymEndsPCNNConfig(), AttentionSynonymEndsPCNN()),
            (AttentionSynonymEndsCNNConfig(), AttentionSynonymEndsCNN()),
            (AttentionFramesCNNConfig(), AttentionFramesCNN()),
            (AttentionFramesPCNNConfig(), AttentionFramesPCNN())]