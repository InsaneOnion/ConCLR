from fastai.vision import *


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

    def _clr_nomatrify(self, embed1, embed2, labels1, labels2, loss_name, record=True):
        embed = torch.cat((embed1, embed2), dim=1)
        labels = torch.cat((labels1, labels2), dim=1)
        loss = torch.tensor(0.0, device="cuda")
        for embed_t, labels_t in zip(embed, labels):
            # embed_t: (catlength, embed_dim)
            # labels_t: (catlength)
            loss_m = torch.tensor(0.0, device="cuda")
            for m in range(labels_t.shape[0]):
                pos_indices = torch.where(
                    (labels_t == labels_t[m])
                    & (torch.arange(labels_t.shape[0], device="cuda") != m)
                )[0]
                pos_logit = embed_t[pos_indices]
                if pos_logit.shape[0] == 0:
                    continue
                a_m = torch.where(torch.arange(labels_t.shape[0], device="cuda") != m)[
                    0
                ]
                s = (embed_t[m] @ embed_t[a_m].T) / self.temprature
                s_max = s.max()  # prevent overflow
                loss_p = torch.tensor(0.0, device="cuda")
                esum = torch.tensor(0.0, device="cuda")
                for a in a_m:
                    esum += torch.exp(
                        torch.dot(embed_t[m], embed_t[a]) / self.temprature - s_max
                    )

                for p in pos_indices:
                    loss_p -= (
                        torch.dot(embed_t[m], embed_t[p]) / self.temprature - s_max
                    )

                    loss_p += torch.log(esum)
                loss_m += loss_p / torch.tensor(pos_logit.size(0), device="cuda")
            loss += loss_m
        loss = loss / embed.size(0)
        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss
        return loss

    def _clr_loss(self, embed1, embed2, labels1, labels2, loss_name, record=True):
        embed = torch.cat((embed1, embed2), dim=1)
        labels = torch.cat((labels1, labels2), dim=1)

        _, max_length = labels.shape
        s = torch.div(
            embed @ embed.transpose(2, 1), self.temprature
        )  # calculate similarity
        am = (
            ~torch.eye(max_length, dtype=torch.bool)
            .unsqueeze(0)
            .expand(*s.shape)
            .cuda()
        )  # negative mask(include pm)
        pm = (
            labels[:, :, None]
            == labels[:, None, :]  # expand label and compare it with its transpose
        ) & am  # positive mask

        # exclude eos
        # p_num = torch.where(
        #     labels.unsqueeze(2) != 0, pm.sum(dim=1).unsqueeze(2), 0
        # )  # count pos num of each m

        # include eos
        p_num = pm.sum(dim=1).unsqueeze(2)  # count pos num of each m

        em = p_num.bool()  # deal with no positive

        s = torch.masked_fill(s, ~em, 0)  # mask no pos
        p = torch.masked_fill(s, ~pm, 0)
        a = torch.logsumexp(s.masked_fill(~am, 0), dim=2)
        a = torch.masked_fill(a, ~em.squeeze(), 0)

        smx = torch.where(
            p != 0,
            p - a.unsqueeze(2),
            torch.tensor(0.0, dtype=torch.float32, device="cuda"),
        )
        dv = torch.where(
            em.squeeze(),
            -torch.sum(smx, dim=2) / p_num.squeeze(),
            torch.tensor(0.0, dtype=torch.float32, device="cuda"),
        )
        l_pair = torch.sum(dv, dim=1)
        loss = torch.mean(l_pair, dim=0)

        if record and loss_name is not None:
            self.losses[f"{loss_name}_loss"] = loss

        return loss

    def forward(self, X, Y, *args):
        self.losses = {}
        features, loss_name = X.get("proj"), X.get("name")
        loss = self._clr_loss(
            features[0],
            features[1],
            torch.argmax(Y[0], dim=2),
            torch.argmax(Y[1], dim=2),
            loss_name,
            *args,
        )
        return loss


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
            clr_loss = self.clr_loss(
                outputs[1][0], gt_labels[-2:], gt_lengths[-2:], *args
            )
            # print("rec_loss", rec_loss)
            # print("clr_loss", 0.2 * clr_loss)
            return rec_loss + self.alpha * clr_loss
        else:
            rec_loss = self.rec_loss(outputs, gt_labels, gt_lengths, *args)
            return rec_loss


if __name__ == "__main__":
    import time

    num_samples = 384
    embed_dim = 512
    max_length = 52
    embed1 = torch.randn(num_samples, max_length, embed_dim, device="cuda")
    embed2 = torch.randn(num_samples, max_length, embed_dim, device="cuda")
    labels1 = torch.randint(0, 10, (num_samples, max_length), device="cuda")
    labels2 = torch.randint(0, 10, (num_samples, max_length), device="cuda")

    # Initialize the loss functions
    contrastive_loss = ContrastiveLoss(temprature=2)

    # Test _clr_nomatrify
    start_time = time.time()
    loss_nomatrify = contrastive_loss._clr_nomatrify(
        embed1, embed2, labels1, labels2, "nomatrify", False
    )
    end_time = time.time()
    time_nomatrify = end_time - start_time
    print(f"_clr_nomatrify Loss: {loss_nomatrify}, Time: {time_nomatrify:.4f} seconds")
    print(f"deviation: {abs(loss_nomatrify-time_nomatrify)}")

    # Test _clr_loss
    start_time = time.time()
    loss_clr = contrastive_loss._clr_loss(
        embed1, embed2, labels1, labels2, "clr", False
    )
    end_time = time.time()
    time_clr = end_time - start_time
    print(f"_clr_loss Loss: {loss_clr}, Time: {time_clr:.4f} seconds")

    # _clr_nomatrify Loss: 2982.53466796875, Time: 100.5598 seconds
    # _clr_loss Loss: 2982.53515625, Time: 0.0468 seconds
    # deviation: 0.00048828125
