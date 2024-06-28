import logging
import torch.nn as nn
from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTranformer
from modules.model import Model
from modules.resnet import resnet45


class ProjectionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, use_bn=True, use_act=True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channel, out_channel)
        self.use_bn = use_bn
        self.use_act = use_act
        if self.use_bn:
            self.norm = nn.BatchNorm1d(out_channel)
        if self.use_act:
            self.act = nn.Sigmoid()

    def forward(self, x):
        embed = self.fc(x)
        if self.use_bn:
            embed = (
                self.norm(embed.permute(0, 2, 1).contiguous())
                .permute(0, 2, 1)
                .contiguous()
            )
        if self.use_act:
            embed = self.act(embed)
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

        self.projection_head = ProjectionLayer(self.out_channels, 512)

    def forward(self, images, *args):
        rec_out, clr_out = [[] for _ in range(2)]
        name = ["vision", "vision_aug1", "vision_aug2"]

        if isinstance(images, (tuple, list)):
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
