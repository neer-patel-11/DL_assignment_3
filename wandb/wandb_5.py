def run_label_smoothing(smoothing):
    wandb.init(
        project="transformer-assignment",
        group="3_5_label_smoothing",
        name=f"smoothing_{smoothing}"
    )

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = NoamScheduler(optimizer, 256, 4000)

    loss_fn = LabelSmoothingLoss(
        vocab_size=TGT_VOCAB_SIZE,
        pad_idx=1,
        smoothing=smoothing
    )

    for epoch in range(15):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss
        })

    wandb.finish()


run_label_smoothing(0.1)
run_label_smoothing(0.0)