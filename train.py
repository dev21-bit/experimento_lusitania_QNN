import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pandas as pd
import time
from pathlib import Path
import random

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# =========================
# CONFIG
# =========================
SEED = 42
CSV_PATH = "mental_state.csv"
LABEL_COL = "Label"

N_QUBITS = 6        # 🔥 reducido (clave performance)
N_LAYERS = 1        # 🔥 reducido
N_CLASSES = 3

MAX_SAMPLES = 400
BATCH_SIZE = 32
KFOLDS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# REPRODUCIBILIDAD
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device_count = torch.cuda.device_count()

print(f"CUDA disponible: {torch.cuda.is_available()} | GPUs: {device_count}")

# =========================
# DATA
# =========================
print("Cargando datos...")
df = pd.read_csv(CSV_PATH)

df, _ = train_test_split(
    df,
    train_size=MAX_SAMPLES,
    stratify=df[LABEL_COL],
    random_state=SEED
)

X = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X_scaled)

# normalización angular
X_angles = 2 * np.pi * (
    (X_reduced - X_reduced.min(axis=0)) /
    (X_reduced.max(axis=0) - X_reduced.min(axis=0) + 1e-8)
) - np.pi

X_tensor = torch.tensor(X_angles, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print("Dataset:", X_tensor.shape)

# =========================
# CIRCUITO CUÁNTICO (OPTIMIZADO)
# =========================
def create_vqc(n_qubits):
    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("θ", n_qubits * 3 * N_LAYERS)

    qc = QuantumCircuit(n_qubits)

    # encoding
    for i in range(n_qubits):
        qc.ry(x[i], i)

    idx = 0
    for _ in range(N_LAYERS):
        for q in range(n_qubits):
            qc.rx(theta[idx], q)
            qc.ry(theta[idx+1], q)
            qc.rz(theta[idx+2], q)
            idx += 3

        # entanglement ligero (más barato que full ring)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    observables = [
        SparsePauliOp("I"*i + "Z" + "I"*(n_qubits-i-1))
        for i in range(n_qubits)
    ]

    return qc, x, theta, observables


qc, input_params, weight_params, observables = create_vqc(N_QUBITS)

estimator = StatevectorEstimator()

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=input_params,
    weight_params=weight_params,
    estimator=estimator
)

quantum_layer = TorchConnector(qnn)

# =========================
# MODELO
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = quantum_layer
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES)
        )

    def freeze_quantum(self):
        for p in self.quantum.parameters():
            p.requires_grad = False

    def unfreeze_quantum(self):
        for p in self.quantum.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.quantum(x)
        return self.classifier(x)

# =========================
# UTILIDADES
# =========================
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")

# =========================
# ENTRENAMIENTO
# =========================
def train_model(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | loss={total_loss:.4f} acc={acc:.4f} f1={f1:.4f}")

# =========================
# K-FOLD PARALELO POR GPU
# =========================
def run_fold(fold_id, train_idx, val_idx, gpu_id=0):

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id % device_count}")
    else:
        device = torch.device("cpu")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    model = HybridModel()

    print(f"\nFold {fold_id} en GPU {gpu_id}")
    train_model(model, train_loader, val_loader, device)

    acc, f1 = evaluate(model, val_loader, device)
    return acc, f1

# =========================
# MAIN K-FOLD
# =========================
print("\nIniciando K-Fold...")

skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

results = []

for i, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor)):

    gpu_id = i % max(device_count, 1)

    acc, f1 = run_fold(i, train_idx, val_idx, gpu_id)
    results.append((acc, f1))

# =========================
# RESULTADOS
# =========================
accs = [r[0] for r in results]
f1s = [r[1] for r in results]

print("\n====================")
print("RESULTADOS FINALES")
print("====================")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")