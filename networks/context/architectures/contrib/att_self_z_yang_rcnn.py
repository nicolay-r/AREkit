from arekit.networks.attention.architectures.self_z_yang import self_attention_by_z_yang
from arekit.networks.context.architectures.att_self_rcnn import AttentionSelfRCNN


class AttentionSelfZYangRCNN(AttentionSelfRCNN):

    def get_attention_alphas(self, rnn_outputs):
        _, alphas = self_attention_by_z_yang(rnn_outputs, 100, return_alphas=True)
        return alphas
