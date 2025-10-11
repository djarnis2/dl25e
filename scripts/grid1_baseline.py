# scripts/grid1_baseline.py
import argparse, itertools, numpy as np, pandas as pd
from pathlib import Path
from src.nn import FullyConnectedNN

# (valgfrit) MedMNIST loader for at gøre scriptet selvkørende
def load_data(sample_train=5000, sample_val=500, seed=0):
    try:
        from medmnist import BloodMNIST
    except ImportError as e:
        raise SystemExit("Installér medmnist: pip install medmnist") from e

    train = BloodMNIST(split="train", download=True, size=28)
    val   = BloodMNIST(split="val",   download=True, size=28)

    Xtr, ytr = train.imgs, train.labels.flatten()
    Xva, yva = val.imgs,   val.labels.flatten()

    rng = np.random.default_rng(seed)
    tr_idx = rng.permutation(len(Xtr))[:sample_train]
    va_idx = np.arange(len(Xva))[:sample_val]

    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    Xva, yva = Xva[va_idx], yva[va_idx]

    # reshape + standardisering (mean/std fra TRAIN!):
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    Xva = Xva.reshape(len(Xva), -1).astype(np.float32)
    mean = Xtr.mean(axis=0); std = Xtr.std(axis=0) + 1e-7
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    return Xtr, ytr, Xva, yva

def count_params(layers):
    total = 0
    for i in range(1, len(layers)):
        total += layers[i-1]*layers[i] + layers[i]
    return total

def main(args):
    np.random.seed(42)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # load data
    X_train, y_train, X_val, y_val = load_data(args.sample_train, args.sample_val, seed=0)

    losses         = ['softmax', 'hinge']
    learning_rates = [0.001, 0.005, 0.01]
    hidden_sizes   = [128, 256, 512]
    reg_strengths  = [1e-4, 1e-3, 1e-2]
    optimizers     = ['adam', 'sgd', 'momentum']

    results = []
    for loss, opt, lr, h, reg in itertools.product(losses, optimizers, learning_rates, hidden_sizes, reg_strengths):
        layers = [X_train.shape[1], h, 8]
        model = FullyConnectedNN(layers=layers, reg_strength=reg, loss=loss, seed=42)

        hist = model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=lr, optimizer=opt
        )

        A_val, _ = model.forward(X_val, training=False)
        y_val_pred = A_val.argmax(axis=1)
        val_acc = float((y_val_pred == y_val).mean())

        final_val_loss  = float(hist['val_loss'][-1])  if len(hist.get('val_loss',  [])) else np.nan
        final_train_loss= float(hist['train_loss'][-1])if len(hist.get('train_loss',[])) else np.nan

        row = {
            'loss': loss, 'optimizer': opt, 'lr': lr, 'hidden': h, 'reg': reg,
            'params': count_params(layers),
            'val_acc': val_acc,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss
        }
        results.append(row)
        # skriv løbende (så man har noget hvis kørslen afbrydes)
        pd.DataFrame(results).to_csv(outdir / "grid1_partial.csv", index=False)
        print(f"{loss:8s} | {opt:8s} | lr={lr:<6} hidden={h:<3} reg={reg:<6} → "
              f"val_acc={val_acc:.3f}, val_loss={final_val_loss:.4f}")

    df = pd.DataFrame(results).sort_values(['val_acc','final_val_loss'], ascending=[False, True])
    print(df.head(10))
    df.to_csv(outdir / "grid1_softmax_vs_hinge.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sample_train", type=int, default=5000)
    ap.add_argument("--sample_val", type=int, default=500)
    ap.add_argument("--outdir", type=str, default="outputs/grid1")
    args = ap.parse_args()
    main(args)
