import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if label.dim() == 0:
        label = label.unsqueeze(0)
    else:
        label = label.view(-1)
    return F.cross_entropy(logits.unsqueeze(0), label)


def bag_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    log_p = F.log_softmax(student_logits / temperature, dim=-1).unsqueeze(0)
    q = F.softmax(teacher_logits / temperature, dim=-1).unsqueeze(0)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def attention_kl_loss(student_attn_logits: torch.Tensor, teacher_guided_attn: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    log_p = F.log_softmax(student_attn_logits / temperature, dim=0)
    q = F.softmax(teacher_guided_attn / temperature, dim=0)
    return F.kl_div(log_p, q, reduction="batchmean") * (temperature ** 2)


def hard_topk_instance_loss(
    feats: torch.Tensor,
    teacher_guided_attn: torch.Tensor,
    inst_classifier: torch.nn.Module,
    k_sample: int = 4,
) -> torch.Tensor:
    k = min(k_sample, teacher_guided_attn.numel() // 2)
    if k < 1:
        return torch.tensor(0.0, device=feats.device)

    top_idx = torch.topk(teacher_guided_attn, k).indices
    bot_idx = torch.topk(-teacher_guided_attn, k).indices
    selected = torch.cat([feats[top_idx], feats[bot_idx]], dim=0)
    targets = torch.cat(
        [torch.ones(k, device=feats.device), torch.zeros(k, device=feats.device)]
    ).long()
    return F.cross_entropy(inst_classifier(selected), targets)


def patch_kd_loss(layer_tokens, teacher_layers, coords_5x, coords_20x, kd_layers):
    device = coords_5x.device
    x5 = int(coords_5x[0].item())
    y5 = int(coords_5x[1].item())
    h = list(layer_tokens.values())[0].shape[1]
    w = list(layer_tokens.values())[0].shape[2]
    token_size_20x = 64
    total_loss = 0.0
    layer_count = 0

    for layer_id in kd_layers:
        if layer_id not in layer_tokens or layer_id not in teacher_layers:
            continue
        s_tokens = layer_tokens[layer_id][0]
        t_cls_seq = teacher_layers[layer_id]
        loss_sum = 0.0
        used = 0

        for j in range(t_cls_seq.shape[0]):
            x20 = int(coords_20x[j, 0].item())
            y20 = int(coords_20x[j, 1].item())
            dx20 = x20 - x5
            dy20 = y20 - y5
            if dx20 < 0 or dy20 < 0 or dx20 >= 1024 or dy20 >= 1024:
                continue
            tx0 = dx20 // token_size_20x
            ty0 = dy20 // token_size_20x
            if tx0 < 0 or ty0 < 0 or tx0 + 4 > w or ty0 + 4 > h:
                continue
            s_mean = s_tokens[ty0 : ty0 + 4, tx0 : tx0 + 4, :].mean(dim=(0, 1))
            loss_sum = loss_sum + F.mse_loss(s_mean, t_cls_seq[j].to(device))
            used += 1

        if used > 0:
            total_loss = total_loss + loss_sum / used
            layer_count += 1

    if layer_count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / layer_count


def attention_weighted_feature_loss(student_final, teacher_final, teacher_attn):
    teacher_final = teacher_final.to(student_final.device)
    weights = teacher_attn.to(student_final.device).float()
    if weights.sum() <= 0:
        weights = torch.ones_like(weights) / weights.numel()
    else:
        weights = weights / weights.sum()
    mse = ((teacher_final - student_final.unsqueeze(0)) ** 2).mean(dim=1)
    return (weights * mse).sum()
