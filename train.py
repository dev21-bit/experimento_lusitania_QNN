import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
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

MAX_SAMPLES = 2000
N_QUBITS = 6
N_LAYERS = 1
N_CLASSES = 3

BATCH_SIZE = 32
EPOCHS_CLASSIC = 8
EPOCHS_FINE = 5
KFOLDS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPUs:", torch.cuda.device_count())

# =========================
# DATA
# =========================
df = pd.read_csv(CSV_PATH)
df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)

X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y = LabelEncoder().fit_transform(df[LABEL_COL])

X = StandardScaler().fit_transform(X)
X = PCA(n_components=N_QUBITS).fit_transform(X)

X = 2*np.pi*(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8) - np.pi

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# =========================
# QUANTUM DEVICE
# =========================
dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs, weights):

    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)

        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# =========================
# MODEL HÍBRIDO
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔥 parámetros cuánticos entrenables
        self.q_params = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 2) * 0.1
        )

        # 🔥 bloque clásico previo (feature learning)
        self.pre_classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Linear(32, N_QUBITS)
        )

        # 🔥 clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES)
        )

    def quantum_layer(self, x):

        # ✔ FIX estable tensorizado
        return torch.stack([
            torch.stack(quantum_circuit(x[i], self.q_params))
            for i in range(x.shape[0])
        ])

    def forward(self, x):

        x = self.pre_classifier(x)     # clásico
        x = self.quantum_layer(x)      # cuántico
        x = self.classifier(x)         # clásico final

        return x

# =========================
# TRAIN STEP
# =========================
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)

        loss.backward()
        opt.step()

        total += loss.item()

    return total

# =========================
# EVAL
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
# K-FOLD + FINE TUNING
# =========================
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

results = []

for fold, (tr, va) in enumerate(skf.split(X, y)):

    print("\n====================")
    print(f"FOLD {fold}")
    print("====================")

    train_loader = DataLoader(
        TensorDataset(X[tr], y[tr]),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X[va], y[va]),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = HybridModel().to(device)

    # =========================
    # FASE 1: ENTRENAMIENTO CLÁSICO
    # =========================
    print("Fase 1: clásico")

    opt = torch.optim.Adam(
        model.pre_classifier.parameters(),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_CLASSIC):
        loss = train_epoch(model, train_loader, opt, loss_fn)
        acc, f1 = evaluate(model, val_loader)
        print(f"[Classic] epoch {epoch+1} loss={loss:.3f} acc={acc:.3f} f1={f1:.3f}")

    # =========================
    # FASE 2: FINE TUNING CUÁNTICO
    # =========================
    print("Fase 2: quantum fine-tuning")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(EPOCHS_FINE):
        loss = train_epoch(model, train_loader, opt, loss_fn)
        acc, f1 = evaluate(model, val_loader)
        print(f"[Quantum FT] epoch {epoch+1} loss={loss:.3f} acc={acc:.3f} f1={f1:.3f}")

    acc, f1 = evaluate(model, val_loader)
    results.append((acc, f1))

# =========================
# RESULTADO FINAL
# =========================
accs = [r[0] for r in results]
f1s = [r[1] for r in results]

print("\n====================")
print("RESULTADO FINAL")
print("====================")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
