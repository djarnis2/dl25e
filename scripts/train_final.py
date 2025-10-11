import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from medmnist import BloodMNIST
from src.nn import FullyConnectedNN

# --- Load data ---
train = BloodMNIST(split="train", download=True, size=28)
val   = BloodMNIST(split="val",   download=True, size=28)
test  = BloodMNIST(split="test",  download=True, size=28)

X_train, y_train = train.imgs, train.labels.flatten()
X_val,   y_val   = val.imgs,   val.labels.flatten()
X_test,  y_test  = test.imgs,  test.labels.flatten()

# subsample (samme som i opgaven)
num_training, num_val = 5000, 500
X_train, y_train = X_train[:num_training], y_train[:num_training]
X_val,   y_val   = X_val[:num_val],       y_val[:num_val]

# reshape & standardize (brug kun train mean/std!)
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_val   = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-7
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

# --- Model (best config) ---
layers = [X_train.shape[1], 512, 256, 8]
nn = FullyConnectedNN(layers=layers, reg_strength=3e-4, loss='softmax',
                      seed=42, dropout_rate=0.2)

hist = nn.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=30, batch_size=128,
    learning_rate=5e-4, optimizer='adam',
    lr_decay_steps=10, lr_decay_factor=0.5
)

# --- Plots ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(hist['train_loss'], label='Train loss')
plt.plot(hist['val_loss'], label='Val loss')
plt.title('Final training (dropout + LR decay)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(hist['val_acc'], label='Val accuracy', color='green')
plt.title('Validation accuracy over epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('final_loss_acc.png', dpi=200)

# --- Eval på test ---
y_pred = nn.predict(X_test)
test_acc = (y_pred == y_test).mean()
val_acc_last = hist['val_acc'][-1] if hist['val_acc'] else float('nan')
print(f"Validation accuracy: {val_acc_last:.4f}")
print(f"Test accuracy:       {test_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = list(train.info['label'].values())

plt.figure(figsize=(7,6))
plt.imshow(cm, cmap='viridis')
plt.title('Confusion matrix (test)')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(range(8), labels, rotation=45, ha='right')
plt.yticks(range(8), labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=8)
plt.tight_layout(); plt.savefig('confusion_matrix_test.png', dpi=200)
