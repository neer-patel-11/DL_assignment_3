import matplotlib.pyplot as plt

def log_attention():
    wandb.init(
        project="transformer-assignment",
        group="3_3_attention_heads",
        name="attention_maps"
    )

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(device)
    model.eval()

    src_batch, _ = next(iter(test_loader))
    src_batch = src_batch.to(device)

    src_mask = make_src_mask(src_batch)

    with torch.no_grad():
        enc_out = model.encode(src_batch, src_mask)

        # grab last encoder layer attention
        layer = model.encoder.layers[-1]
        attn_module = layer.self_attn

        Q = attn_module.W_q(enc_out)
        K = attn_module.W_k(enc_out)

        Q = Q.view(1, -1, attn_module.num_heads, attn_module.d_k).transpose(1,2)
        K = K.view(1, -1, attn_module.num_heads, attn_module.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1))
        attn = torch.softmax(scores, dim=-1)

        for h in range(attn.size(1)):
            plt.imshow(attn[0,h].cpu())
            wandb.log({f"head_{h}": wandb.Image(plt)})
            plt.clf()

    wandb.finish()


log_attention()