"""
SE-ResNet, SE_ResNeXt codes are gently borrowed from
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from senet import se_resnext101_32x4d


class BaseNetwork(nn.Module):
    """ Load Pretrained Module """

    def __init__(self, model_name, embedding_dim, feature_extracting, use_pretrained):
        super(BaseNetwork, self).__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.feature_extracting = feature_extracting
        self.use_pretrained = use_pretrained

        self.model_ft = initialize_model(self.model_name,
                                         self.embedding_dim,
                                         self.feature_extracting,
                                         self.use_pretrained)

    def forward(self, x):
        out = self.model_ft(x)
        return out


class SelfAttention(nn.Module):
    """ Self attention Layer
    https://github.com/heykeetae/Self-Attention-GAN"""

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class EmbeddingNetwork(BaseNetwork):
    """ Wrapping Modules to the BaseNetwork """

    def __init__(self, model_name, embedding_dim, feature_extracting, use_pretrained,
                 attention_flag=False, cross_entropy_flag=False, edge_cutting=False):
        super(EmbeddingNetwork, self).__init__(model_name, embedding_dim, feature_extracting, use_pretrained)
        self.attention_flag = attention_flag
        self.cross_entropy_flag = cross_entropy_flag
        self.edge_cutting = edge_cutting

        self.model_ft_convs = nn.Sequential(*list(self.model_ft.children())[:-1])
        self.model_ft_embedding = nn.Sequential(*list(self.model_ft.children())[-1:])

        if self.attention_flag:
            if self.model_name == 'densenet161':
                self.attention = SelfAttention(2208, 'relu')
            elif self.model_name == 'resnet101':
                self.attention = SelfAttention(2048, 'relu')
            elif self.model_name == 'inceptionv3':
                self.attention = SelfAttention(2048, 'relu')
            elif self.model_name == 'seresnext':
                self.attention = SelfAttention(2048, 'relu')

        if self.cross_entropy_flag:
            self.fc_cross_entropy = nn.Linear(self.model_ft.classifier.in_features, 1000)

    def forward(self, x):
        x = self.model_ft_convs(x)
        x = F.relu(x, inplace=True)

        if self.attention_flag:
            x = self.attention(x)

        if self.edge_cutting:
            x = F.adaptive_avg_pool2d(x[:, :, 1:-1, 1:-1], output_size=1).view(x.size(0), -1)
        else:
            x = F.adaptive_avg_pool2d(x, output_size=1).view(x.size(0), -1)
            # x = gem(x).view(x.size(0), -1)
        out_embedding = self.model_ft_embedding(x)

        if self.cross_entropy_flag:
            out_cross_entropy = self.fc_cross_entropy(x)
            return out_embedding, out_cross_entropy
        else:
            return out_embedding


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, embedding_dim, feature_extracting, use_pretrained=True):
    if model_name == "densenet161":
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_features, embedding_dim)
    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, embedding_dim)
    elif model_name == "inceptionv3":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, embedding_dim)
    elif model_name == "seresnext":
        model_ft = se_resnext101_32x4d(num_classes=1000)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_features, embedding_dim)
    else:
        raise ValueError

    return model_ft


# GeM Pooling
def gem(x, p=3, eps=1e-6):
    return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), output_size=1).pow(1. / p)
