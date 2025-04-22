import math
import os
import torch
from torch import nn
import torch.nn.functional as F

from network.DSwin import AdaSwinTransformer, SwinTransformer_Teacher

from network.BatchBranch import Batch_GNN


def pdist(vectors):
    """
    vectors: (batch, 2048)
    distance: EuclideanDistance
    The more similar two images are, the higher the similarity is and the smaller the distance is
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def create_global_branch(cfg: dict, only_teacher_model: bool = False):

    embed_dim = 128
    depth = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    pruning_loc = cfg["pruning_loc"]
    keep_rate = cfg["keep_rate"]
    pretrained_path = cfg['pretrain_path']


    if only_teacher_model:
        teacher_model = SwinTransformer_Teacher(
            img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"],
            embed_dim=embed_dim, depths=depth, num_heads=num_heads,
        )
        checkpoint = torch.load(pretrained_path, map_location="cpu")["model"]
        teacher_model.load_state_dict(checkpoint, strict=False)
        return teacher_model
    else:

        model = AdaSwinTransformer(
            img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"],
            embed_dim=embed_dim, depths=depth, num_heads=num_heads,
            keep_rate=keep_rate, pruning_loc=pruning_loc
        )
        global_feature_dim = model.num_features
        if cfg["load_pretrained"]:
            checkpoint = torch.load(pretrained_path, map_location="cpu")["model"]
            model.load_state_dict(checkpoint, strict=False)
        num_patches = model.layers[-1].input_resolution[0] * model.layers[-1].input_resolution[1]  # type: ignore
        return model, pruning_loc, global_feature_dim

class WeightLearningModule(nn.Module):
    def __init__(self, input_dim):
        super(WeightLearningModule, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, global_features, local_features):
        combined_feats = torch.cat([global_features, local_features], dim=1)
        weights = self.fc(combined_feats)
        alpha, beta = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        return alpha, beta


class GCCR(nn.Module):

    def __init__(self, model_cfg, local_cfg, NoBatch_GNN, NoAWF, NoRank_loss):
        super(GCCR, self).__init__()

        self.NoBatch_GNN = NoBatch_GNN
        self.NoAWF = NoAWF
        self.NoRank_loss = NoRank_loss

        self.keep_rate = model_cfg["keep_rate"]

        self.has_cls_token = False

        self.backbone, self.pruning_loc, global_feature_dim = create_global_branch(cfg={
                **model_cfg,
            })  # type: ignore

        GNN_channels = 1024
        self.feature_stage = 4 #
        self.Batch_Branch = Batch_GNN(GNN_channels, batch_size=local_cfg["batch_size"],depth=local_cfg["depth"] )
        self.local_proj_layer = nn.Linear(GNN_channels, global_feature_dim)

        self.weight_learning_module = WeightLearningModule(2*global_feature_dim)

        # ### feature interaction module

        ### classifier
        # self.classifier = Mlp(global_feature_dim, global_feature_dim * 4, model_cfg["num_classes"])
        self.classifier = nn.Linear(global_feature_dim, model_cfg["num_classes"])

        self.softmax_layer = nn.LogSoftmax(dim=1)

    def get_param_groups(self):
        param_groups = [[], []]  # backbone, otherwise
        for name, param in super().named_parameters():
            if name.startswith("backbone") and "score_predictor" not in name:
                param_groups[0].append(param)
            else:
                param_groups[1].append(param)
        return param_groups

    def fuse_feature(self, global_feature, local_feature=None, NoAWF=False):
        """ fuse global feature with local feature
        global feature: produced by backbone
        local feature: produced by local branch
        """
        alpha, beta = torch.ones(1), torch.ones(1)
        feature = global_feature
        if local_feature is not None:
            ### addition
            local_feature = self.local_proj_layer(local_feature)
            # feature = feature + local_feature
            if not NoAWF:
                alpha, beta = self.weight_learning_module(global_feature, local_feature)
                feature = alpha * global_feature + beta * local_feature
            else:
                feature = global_feature + local_feature

        return feature

    def forward(self, images, targets=None, flag=None):
        ### global features
        # global_features:B,C global_patch_features:B,N,C
        global_features, global_patch_features, global_attentions, decision_mask, decision_mask_list, feature_list = self.backbone.forward_features(
            images)

        # # combine global features with local features
        local_features = None
        if not self.NoBatch_GNN:
            local_features = self.Batch_Branch(feature_list[self.feature_stage-1])  # B,C

        features = self.fuse_feature(global_feature=global_features, local_feature=local_features, NoAWF=self.NoAWF)
        logits = self.classifier(features)

        ### global feature interaction
        if self.training and not self.NoRank_loss:
            with torch.no_grad():

                intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(global_features, targets)

            features1_self = torch.cat([features[intra_pairs[:, 0]], features[inter_pairs[:, 0]]], dim=0)
            features2_other = torch.cat([features[intra_pairs[:, 1]], features[inter_pairs[:, 1]]], dim=0)

            # obtain classification probability
            logit1_self = self.classifier(features1_self)
            logit2_other = self.classifier(features2_other)

            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)  # ori samples and ori samples
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)  # intra and inter samples
            self_logits = logit1_self #torch.cat([logit1_self, logit2_self], dim=0)
            other_logits = logit2_other #torch.cat([logit1_other, logit2_other], dim=0)

            # prepare margin rank loss calculation
            self_scores = self.softmax_layer(self_logits)
            self_scores = self_scores[
                torch.arange(self_scores.shape[0]), labels1.to(torch.long)] # self_scores[labels1]

            other_scores = self.softmax_layer(other_logits)
            other_scores = other_scores[
                torch.arange(other_scores.shape[0]), labels2.to(torch.long)]


            return logits, self_scores, other_scores, global_features, decision_mask_list
        else:
            # without feature interaction
            if self.training and self.NoRank_loss:
                return logits, global_features, decision_mask_list
            else:
                #
                return logits

    def get_pairs(self, embeddings, labels):
        """
        embeddings: B,C
        labels: B,1

        """
        distance_matrix = pdist(embeddings)  # Calculate the similarity between images in a batch

        labels = labels.unsqueeze(dim=1)
        batch_size = embeddings.shape[0]
        lb_eqs = (labels == torch.t(labels))

        dist_same = distance_matrix.clone()
        lb_eqs = lb_eqs.fill_diagonal_(fill_value=False,
                                       wrap=False)  # Positions with the same category are True; Positions with different categories are False; Positions with themselves are False
        dist_same[lb_eqs == False] = float(
            "inf")  # Samples of different classes are infinitely far away; The distance between self and self is infinite. There is an effective distance only between samples with the same class
        intra_idxs = torch.argmin(dist_same, dim=1)  # intra; Match similar images within a class

        dist_diff = distance_matrix.clone()
        lb_eqs = lb_eqs.fill_diagonal_(fill_value=True,
                                       wrap=False)  # Same category is True. Different classes are False. Self and its position are True
        dist_diff[lb_eqs == True] = float(
            "inf")  # The distance between samples with the same class is infinite. The distance between oneself and oneself is infinite; There is only an effective distance between samples with different classes
        inter_idxs = torch.argmin(dist_diff, dim=1)  # inter; Match images that are similar between classes

        intra_labels = torch.cat([labels[:], labels[intra_idxs]], dim=1)
        inter_labels = torch.cat([labels[:], labels[inter_idxs]], dim=1)
        intra_pairs = torch.cat(
            [torch.arange(0, batch_size).unsqueeze(dim=1).to(embeddings.device), intra_idxs.unsqueeze(dim=1)], dim=1)
        inter_pairs = torch.cat(
            [torch.arange(0, batch_size).unsqueeze(dim=1).to(embeddings.device), inter_idxs.unsqueeze(dim=1)], dim=1)

        # intra_pairs, inter_pairs
        # intra_labels, inter_labels
        # pairs: Original sample index - Corresponding similar sample index
        return intra_pairs, inter_pairs, intra_labels, inter_labels


