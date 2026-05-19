# 🔥 Backup original function ONCE
original_attn = scaled_dot_product_attention


def no_scale_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1))  # ❌ no sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


def run_scaling(use_scaling=True):
    global scaled_dot_product_attention  # 🔥 IMPORTANT

    wandb.init(
        project="transformer-assignment",
        group="3_2_scaling_ablation",
        name="with_scaling" if use_scaling else "no_scaling"
    )

    # 🔥 Patch BEFORE model creation
    if use_scaling:
        scaled_dot_product_attention = original_attn
    else:
        scaled_dot_product_attention = no_scale_attention

    # ✅ Now build model AFTER patch
    net = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = NoamScheduler(optimizer, 256, 4000)

    loss_fn = LabelSmoothingLoss(TGT_VOCAB_SIZE, pad_idx=1, smoothing=0.1)

    for epoch in range(15):
        train_loss = run_epoch(
            train_loader, net, loss_fn,
            optimizer, scheduler,
            epoch, True, device
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss
        })

    wandb.finish()


# ✅ Run both experiments
run_scaling(True)
run_scaling(False)