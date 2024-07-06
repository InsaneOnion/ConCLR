import logging
import torch.nn as nn
import torch.functional as F
from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTranformer
from modules.model import Model
from modules.resnet import resnet45


class ProjectionHead(nn.Module):
    def __init__(self, in_channel, out_channel, head="mlp") -> None:
        super().__init__()
        if head == "linear":
            self.head = nn.Linear(in_channel, out_channel)
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_channel, in_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_channel, out_channel),
            )

    def forward(self, x):
        embed = F.normalize(self.head(x), dim=1)
        return embed


class ConCLR_Vision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, [1.0, 0.5, 0.5])
        self.out_channels = ifnone(config.model_vision_d_model, 512)
        self.embed_channels = ifnone(config.model_embedding_channels, 512)

        if config.model_vision_backbone == "transformer":
            self.backbone = ResTranformer(config)
        else:
            self.backbone = resnet45()

        if config.model_vision_attention == "position":
            mode = ifnone(config.model_vision_attention_mode, "nearest")
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == "attention":
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )
        else:
            raise Exception(f"{config.model_vision_attention} is not valid.")
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f"Read vision model from {config.model_vision_checkpoint}.")
            self.load(config.model_vision_checkpoint)

        self.projection_head = ProjectionHead(self.out_channels, self.embed_channels)

    def forward(self, images, *args):
        rec_out, clr_out = [[] for _ in range(2)]

        if isinstance(images, (tuple, list)):
            name = ["vision", "vision_aug1", "vision_aug2"]
            for idx, image in enumerate(images):
                features = self.backbone(image)  # (N, E, H, W)
                attn_vecs, attn_scores = self.attention(
                    features
                )  # (N, T, E), (N, T, H, W)
                logits = self.cls(attn_vecs)  # (N, T, C)
                pt_lengths = self._get_length(logits)

                rec_out.append(
                    {
                        "feature": attn_vecs,
                        "logits": logits,
                        "pt_lengths": pt_lengths,
                        "attn_scores": attn_scores,
                        "loss_weight": self.loss_weight[idx],
                        "name": name[idx],
                    }
                )

            aligned_feats_one, aligned_feats_two = [
                ro["feature"] for ro in rec_out[-2:]
            ]

            proj_feats_one = self.projection_head(aligned_feats_one)
            proj_feats_two = self.projection_head(aligned_feats_two)
            proj_feats = (proj_feats_one, proj_feats_two)

            clr_out.append({"proj": proj_feats, "name": "clr"})

            return (rec_out, clr_out)
        else:
            features = self.backbone(images)
            attn_vecs, attn_scores = self.attention(features)
            logits = self.cls(attn_vecs)
            pt_lengths = self._get_length(logits)

            rec_out = {
                "feature": attn_vecs,
                "logits": logits,
                "pt_lengths": pt_lengths,
                "attn_scores": attn_scores,
                "loss_weight": self.loss_weight[0],
                "name": "vision",
            }

            return rec_out
