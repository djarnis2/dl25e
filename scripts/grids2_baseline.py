import itertools, numpy as np, pandas as pd
from medmnist import BloodMNIST
from src.nn import FullyConnectedNN

np.random.seed(42)
depths = []
def add(*h): depths.append(h)
# dine arkitekturer:
add(256,128); add(256,256); add(512,256); add(256,256,128)

lrs  = [1e-3, 5e-4]
regs = [1e-4, 3e-4, 1e-3]
drops= [0.0, 0.2, 0.3]   # screening: dropout ikke aktiv i forward (training=False) for fairness

# data
train, val = BloodMNIST(split="train", download=True, size=28), BloodMNIST(split="val", download=True, size=28)
X_train, y_train = train.imgs[:5000].reshape(5000, -1).astype(np.float32), train.labels[:5000].flatten()
X_val,   y_val   = val.imgs[:500].reshape(500, -1).astype(np.float32),     val.labels[:500].flatten()
mean, std = X_train.mean(axis=0), X_train.std(axis=0)+1e-7
X_train, X_val = (X_train-mean)/std, (X_val-mean)/std

rows=[]
for hs, lr, reg, dr in itertools.product(depths, lrs, regs, drops):
    layers=[X_train.shape[1], *hs, 8]
    model=FullyConnectedNN(layers=layers, reg_strength=reg, loss='softmax', seed=42, dropout_rate=dr)
    hist=model.fit(X_train,y_train,X_val=X_val,y_val=y_val,epochs=12,batch_size=128,learning_rate=lr,optimizer='adam')
    val_logits,_=model.forward(X_val, training=False)  # dropout off in screening
    val_acc=float((val_logits.argmax(1)==y_val).mean())
    rows.append({'layers':str(list(hs)),'lr':lr,'reg':reg,'dropout':dr,'val_acc':val_acc,
                 'final_val_loss':float(hist['val_loss'][-1]) if hist['val_loss'] else np.nan})
    print(f"[{', '.join(map(str,hs))}] lr={lr} reg={reg} dr={dr} → val_acc={val_acc:.3f}")

pd.DataFrame(rows).to_csv("grid_deep_screening.csv", index=False)
