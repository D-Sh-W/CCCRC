import torch.nn as nn
import torch


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)

        self.value = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):

        if Q_feature.dim() == 2:
            Q_feature = Q_feature.unsqueeze(1)
        if K_feature.dim() == 2:
            K_feature = K_feature.unsqueeze(1)

        B, N, C = Q_feature.shape

        Q = self.query(Q_feature)

        K = self.key(K_feature)

        V = self.value(K_feature)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(C, dtype=torch.float32).to(Q.device))

        attention_weights = self.softmax(attention_scores)

        attended_features = torch.matmul(attention_weights, V)

        return attended_features.squeeze(1)
