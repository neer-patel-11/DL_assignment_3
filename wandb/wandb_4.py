import torch.nn as nn

class LearnedPosEnc(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.embed(pos)


def run_positional(use_learned=False):
    wandb.init(
        project="transformer-assignment",
        group="3_4_positional_encoding",
        name="learned" if use_learned else "sinusoidal"
    )

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(device)

    if use_learned:
        model.src_pos_enc = LearnedPosEnc(256).to(device)
        model.tgt_pos_enc = LearnedPosEnc(256).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = NoamScheduler(optimizer, 256, 4000)

    loss_fn = LabelSmoothingLoss(TGT_VOCAB_SIZE, pad_idx=1, smoothing=0.1)

    for epoch in range(15):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "bleu": bleu
        })

    wandb.finish()


run_positional(False)
run_positional(True)