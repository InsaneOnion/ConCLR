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
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def _clr_loss(self, X, Y, idx=None, record=True):
        X, loss_name = X.get("proj"), X.get("name")
        Z = torch.cat(X, dim=1)  # (batch_size, max_length, embedding_dim)
        Y_aug = torch.cat(Y, dim=1)  # (batch_size, max_length, charset_length)

        # Determine indices where characters are not end-of-sequence
        not_end = torch.all(
            Y_aug == Y_aug[:, -1:], dim=2
        ).cuda()  # (batch_size, max_length)

        # Create mask
        eq_indices = (
            torch.any(Y_aug[:, :, None] != Y_aug[:, None, :], dim=3)
            | not_end[:, :, None]
        )

        diag_mask = torch.eye(Y_aug.shape[1], dtype=torch.bool).cuda()
        a_mask = diag_mask[None, :, :].expand_as(eq_indices) | not_end[:, :, None]

        p_mask = eq_indices | a_mask

        s = torch.matmul(Z, Z.transpose(1, 2)) / self.tau
        pm_length = p_mask.sum(dim=2) + 1

        p = torch.logsumexp(s.masked_fill(p_mask, -float("inf")), dim=1)
        a = torch.logsumexp(s.masked_fill(a_mask, -float("inf")), dim=1)
        loss = torch.mean(
            -torch.sum(
                (p.masked_fill(p == -float("inf"), 1e-9) - a) / pm_length,
                dim=1,
            ),
            dim=0,
        )
        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss
        return loss

    def forward(self, X, Y, Y_length, *args):
        self.losses = {}
        return self._clr_loss(X[0], Y, Y_length, *args)


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
