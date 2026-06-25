import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

import pennylane as qml

# =========================
# CONFIG
# =========================
SEED = 42
CSV_PATH = "mental_state.csv"
LABEL_COL = "Label"

N_QUBITS = 6
N_LAYERS = 1
N_CLASSES = 3

MAX_SAMPLES = 400
BATCH_SIZE = 32
KFOLDS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPUs:", torch.cuda.device_count())

# =========================
# DATA
# =========================
df = pd.read_csv(CSV_PATH)

df, _ = train_test_split(
    df,
    train_size=MAX_SAMPLES,
    stratify=df[LABEL_COL],
    random_state=SEED
)

X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y = LabelEncoder().fit_transform(df[LABEL_COL])

X = StandardScaler().fit_transform(X)
X = PCA(n_components=N_QUBITS).fit_transform(X)

# normalización angular
X = 2*np.pi*(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) - np.pi

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)

# =========================
# PENNYLANE DEVICE
# =========================
dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs, weights):

    # encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # variational layers
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)

        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    # IMPORTANT: return tensor-friendly structure
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# =========================
# HYBRID MODEL (FIXED)
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_params = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 2) * 0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES)
        )

    def forward(self, x):
        # 🔥 FIX: ensure tensor stacking
        q_out = torch.stack([
            torch.stack(quantum_circuit(x[i], self.q_params))
            for i in range(x.shape[0])
        ])

        return self.classifier(q_out)

# =========================
# TRAIN LOOP
# =========================
def train(model, loader):
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | loss={total_loss:.4f}")

# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)

            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(yb.numpy())

    return (
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro")
    )

# =========================
# K-FOLD
# =========================
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

results = []

for fold, (tr, va) in enumerate(skf.split(X_t, y_t)):

    print("\n====================")
    print("FOLD", fold)
    print("====================")

    train_loader = DataLoader(
        TensorDataset(X_t[tr], y_t[tr]),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_t[va], y_t[va]),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = HybridModel()

    train(model, train_loader)

    acc, f1 = evaluate(model, val_loader)

    print(f"Fold {fold} -> ACC: {acc:.4f} | F1: {f1:.4f}")

    results.append((acc, f1))

# =========================
# FINAL RESULTS
# =========================
accs = [r[0] for r in results]
f1s = [r[1] for r in results]

print("\n====================")
print("RESULTADO FINAL")
print("====================")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")