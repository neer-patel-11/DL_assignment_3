def run_exp(use_noam=True):
    wandb.init(
        project="transformer-assignment",
        group="3_1_noam_vs_fixed",
        name="noam" if use_noam else "fixed_lr"
    )

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = None
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)

    loss_fn = LabelSmoothingLoss(TGT_VOCAB_SIZE, pad_idx=1, smoothing=0.1)

    for epoch in range(13):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        bleu       = evaluate_bleu(model, test_loader, tgt_vocab, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu": bleu
        })

    wandb.finish()


# run_exp(use_noam=True)
run_exp(use_noam=False)