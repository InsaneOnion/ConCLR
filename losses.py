from fastai.vision import *

from modules.model import Model


class MultiLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, idx=None, record=True):
        loss_name = output.get("name")
        pt_logits, weight = output["logits"], output["loss_weight"]

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        nll = output.get("nll")
        if nll is not None:
            loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
        else:
            loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum(
                [self._ce_loss(o, *args) for o in outputs if o["loss_weight"] > 0.0]
            )
        else:
            return self._ce_loss(outputs, *args, record=False)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax:
            log_prob = F.log_softmax(input, dim=-1)
        else:
            log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RecLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, idx=None, record=True):
        loss_name = output.get("name")
        pt_logits, weight = output["logits"], output["loss_weight"]

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        nll = output.get("nll")
        if nll is not None:
            loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
        else:
            loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss
        return loss

    def forward(self, outputs, gt_labels, gt_lengths, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum(
                [
                    self._ce_loss(o, label, length, *args)
                    for o, label, length in zip(outputs, gt_labels, gt_lengths)
                    if o["loss_weight"] > 0.0
                ]
            )
        else:
            return self._ce_loss(outputs, gt_labels, gt_lengths, *args, record=False)


class ContrastiveLoss(nn.Module):
    def __init__(self, temprature=2):
        super().__init__()
        self.temprature = temprature

    def _clr_loss(self, logits1, logits2, labels1, labels2, loss_name, record=True):
        # Concatenate logits and labels
        concatenated_logits = torch.cat((logits1, logits2), dim=1)
        concatenated_labels = torch.cat((labels1, labels2), dim=1)

        b, max_length = concatenated_labels.shape
        # Create attention mask and positive mask
        attention_mask = (
            ~torch.eye(max_length, dtype=torch.bool)
            .unsqueeze(0)
            .expand(b, -1, -1)
            .cuda()
        )  # mask for not considering self-attention

        positive_mask = (
            concatenated_labels[:, :, None] != concatenated_labels[:, None, :]
        ) & attention_mask  # mask for considering positive examples

        # Count number of positive pairs and create mask for no positive pairs
        positive_pairs_count = positive_mask.sum(dim=1).unsqueeze(2)
        no_positive_mask = (
            attention_mask
            & positive_pairs_count.expand(b, max_length, max_length).bool()
        )

        # Calculate similarity matrix
        similarity_matrix = torch.div(
            concatenated_logits @ concatenated_logits.transpose(2, 1), self.temperature
        )
        similarity_matrix = torch.masked_fill(
            similarity_matrix, ~no_positive_mask, 0
        )  # mask no positive pairs
        similarity_matrix = torch.exp(
            similarity_matrix - similarity_matrix.max(dim=2, keepdim=True)[0]
        )  # prevent overflow

        # Apply masks to similarity matrix
        positive_similarity = torch.masked_fill(similarity_matrix, ~positive_mask, 0)
        attention_similarity = torch.masked_fill(similarity_matrix, ~attention_mask, 0)
        attention_similarity = torch.masked_fill(
            attention_similarity, ~no_positive_mask, 0
        ).sum(dim=2)
        attention_similarity = torch.where(
            attention_similarity != 0, torch.log(attention_similarity), 0
        )

        # Calculate contrastive loss
        contrastive_loss_matrix = torch.where(
            positive_similarity != 0,
            positive_similarity - attention_similarity.unsqueeze(2),
            0,
        )
        loss_per_example = torch.sum(
            -torch.sum(contrastive_loss_matrix, dim=2) / positive_pairs_count.squeeze(),
            dim=1,
        )
        loss = torch.mean(loss_per_example, dim=0)

        # Record loss if specified
        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss

        return loss

    def forward(self, X, Y, *args):
        self.losses = {}
        features, loss_name = X[0].get("proj"), X[0].get("name")
        return self._clr_loss(
            features[0],
            features[1],
            torch.argmax(Y[0], dim=2),
            torch.argmax(Y[1], dim=2),
            loss_name,
            *args,
        )


class TotalLosses(nn.Module):
    def __init__(self, one_hot=True, tau=2, alpha=0.2):
        super().__init__()
        self.rec_loss = RecLosses(one_hot)
        self.clr_loss = ContrastiveLoss(tau)
        self.alpha = alpha

    @property
    def last_losses(self):
        return {
            "vision_loss": self.rec_loss.losses["vision_loss"],
            "vision_aug1_loss": self.rec_loss.losses["vision_aug1_loss"],
            "vision_aug2_loss": self.rec_loss.losses["vision_aug2_loss"],
            "rec_loss": self.rec_loss.losses["vision_loss"]
            + self.rec_loss.losses["vision_aug1_loss"]
            + self.rec_loss.losses["vision_aug2_loss"],
            "clr_loss": self.clr_loss.losses["clr_loss"],
        }

    def forward(self, outputs, gt_labels, gt_lengths, *args):
        if isinstance(outputs, (tuple, list)):
            rec_loss = self.rec_loss(outputs[0], gt_labels, gt_lengths, *args)
            clr_loss = self.clr_loss(outputs[1], gt_labels[-2:], gt_lengths[-2:], *args)
            return rec_loss + self.alpha * clr_loss
        else:
            rec_loss = self.rec_loss(outputs, gt_labels, gt_lengths, *args)
            return rec_loss
