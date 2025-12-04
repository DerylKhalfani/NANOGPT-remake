import torch
from model import GPT, GPTConfig
from training import get_batch, block_size  # assumes training.py defines these


def main():
    print("=== TESTING PIPELINE ===")

    # ------------------------
    # 0. Device
    # ------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")

    # ------------------------
    # 1. Test forward pass
    # ------------------------
    print("\n[1] Testing forward pass")
    cfg = GPTConfig(block_size=block_size, vocab_size=50304)
    model = GPT(cfg).to(device)

    # fake batch: (B=2, T=50)
    B, T = 2, 50
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(x, x)
    print("  logits.shape:", logits.shape)   # (2, 50, 50304)
    print("  loss:", loss)                  # scalar

    # ------------------------
    # 2. Test a training step
    # ------------------------
    print("\n[2] Testing single training step")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    logits, loss = model(x, x)
    loss.backward()
    optimizer.step()
    print("  Training step OK")

    # ------------------------
    # 3. Test generate()
    # ------------------------
    print("\n[3] Testing generate()")
    # simple prompt of 3 tokens, on same device as model
    idx = torch.randint(0, cfg.vocab_size, (1, 3), device=device)
    out = model.generate(idx, max_new_tokens=10)
    print("  Generated shape:", out.shape)
    print("  Generated tokens:", out)

    # ------------------------
    # 4. Test get_batch() dataloader
    # ------------------------
    print("\n[4] Testing get_batch()")
    xb, yb = get_batch("train", batch_size=4)
    print("  xb.shape:", xb.shape)   # (4, block_size)
    print("  yb.shape:", yb.shape)
    print("  xb.dtype:", xb.dtype)
    print("  yb.dtype:", yb.dtype)

    print("\nALL TESTS COMPLETED SUCCESSFULLY ✅")


if __name__ == "__main__":
    main()
