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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

import pennylane as qml

# =========================
# FIJAR SEMILLAS PARA REPRODUCIBILIDAD
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# CONFIGURACIÓN
# =========================
CSV_PATH   = "mental_state.csv"
LABEL_COL  = "Label"

N_QUBITS   = 10
N_LAYERS   = 2
N_CLASSES  = 3
N_WEIGHTS  = N_QUBITS * 3 * N_LAYERS   # 3 rotaciones × N_QUBITS × N_LAYERS

DEVICE      = "cpu"
KFOLDS      = 3
MAX_SAMPLES = 2479
BATCH_SIZE  = 32

OUTPUT_DIR = Path("resultados_entrenamiento")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(DEVICE)

# =========================
# CARGA Y PREPARACIÓN
# =========================
print("Cargando datos...")
if MAX_SAMPLES < len(df):
    df, _ = train_test_split(
        df,
        train_size=MAX_SAMPLES,
        stratify=df[LABEL_COL],
        random_state=SEED
    )
# Si MAX_SAMPLES >= len(df), usamos todo el dataset directamente
X     = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X_scaled)

# Escalar a [-pi, pi]
min_val  = X_reduced.min(axis=0)
max_val  = X_reduced.max(axis=0)
X_angles = 2 * np.pi * (X_reduced - min_val) / (max_val - min_val + 1e-8) - np.pi

X_tensor = torch.tensor(X_angles, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"Dataset final: {X_tensor.shape}")
print(f"Distribución de clases: {np.bincount(y)}")

# =========================
# DISPOSITIVO CUÁNTICO
# =========================
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print("Simulador: lightning.qubit (C++, rápido)")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("Simulador: default.qubit")

# =========================
# CIRCUITO CUÁNTICO
# =========================
# Se define con @qml.qnode para ser idiomático en PennyLane moderno.
# interface="torch"  → los tensores de entrada/pesos son torch.Tensor
# diff_method="best" → PennyLane elige el mejor método de diferenciación disponible
#                      (parameter-shift si no hay adjoint, adjoint con lightning)
@qml.qnode(dev, interface="torch", diff_method="best")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    """
    VQC equivalente al circuito de Qiskit original:
      - Data encoding : RX + RY por qubit (angle embedding doble)
      - N_LAYERS capas: RX+RY+RZ por qubit + entanglement ring CNOT
      - Observables   : <Z_i> para i en [0, N_QUBITS)   → N_QUBITS salidas reales
    """
    # --- Codificación de datos ---
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i], wires=i)

    # --- Capas variacionales ---
    idx = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[idx],     wires=q)
            qml.RY(weights[idx + 1], wires=q)
            qml.RZ(weights[idx + 2], wires=q)
            idx += 3

        # Entanglement ring (idéntico al cx ring de Qiskit)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    # PennyLane >= 0.38 devuelve tuple de tensores scalares al usar múltiples expval
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS))


# =========================
# CAPA CUÁNTICA COMO nn.Module
# =========================
class QuantumLayer(nn.Module):
    """
    Equivalente a TorchConnector(qnn) de Qiskit.

    Los pesos del circuito son nn.Parameter para que Adam/SGD los actualice
    automáticamente. El gradiente fluye a través de PennyLane vía autograd
    de PyTorch (parameter-shift rule o adjoint differentiation).
    """
    def __init__(self, n_weights: int):
        super().__init__()
        # Inicialización pequeña → evita barren plateaus en el inicio
        init_w = torch.empty(n_weights).uniform_(-np.pi / 4, np.pi / 4)
        self.weights = nn.Parameter(init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantum_circuit devuelve tuple de N_QUBITS tensores scalar por muestra.
        # torch.stack convierte esa tuple en un vector (N_QUBITS,).
        # Procesamos muestra a muestra (batch loop); PennyLane no soporta
        # vmap sobre QNodes en versión estable, así que el loop es necesario.
        # PennyLane (default.qubit / lightning.qubit) devuelve float64.
        # Lo casteamos a float32 para compatibilidad con las capas Linear de PyTorch.
        rows = [
            torch.stack(list(quantum_circuit(x[i], self.weights))).float()
            for i in range(x.shape[0])
        ]
        return torch.stack(rows)   # → (batch, N_QUBITS), float32


# =========================
# MODELO HÍBRIDO
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum    = QuantumLayer(N_WEIGHTS)
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, N_CLASSES)
        )

    def freeze_quantum(self):
        """Fase 1: congela los pesos cuánticos, solo entrena la parte clásica."""
        self.quantum.weights.requires_grad = False

    def unfreeze_quantum(self):
        """Fase 2: descongela para fine-tuning completo."""
        self.quantum.weights.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum(x)        # (batch, N_QUBITS)
        return self.classifier(q_out)  # (batch, N_CLASSES)


# =========================
# EVALUACIÓN
# =========================
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# =========================
# ENTRENAMIENTO EN DOS FASES
# =========================
def train_two_phase(model, train_loader, val_loader, fold, device):
    criterion = nn.CrossEntropyLoss()
    history = {
        'phase1': {'loss': [], 'acc': [], 'time': []},
        'phase2': {'loss': [], 'acc': [], 'time': []}
    }

    # -------- FASE 1: Solo parte clásica --------
    print("\n--- FASE 1 (clásico) ---")
    model.freeze_quantum()
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    for epoch in range(15):
        start = time.time()
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        acc      = evaluate(model, val_loader, device)
        t        = time.time() - start

        history['phase1']['loss'].append(avg_loss)
        history['phase1']['acc'].append(acc)
        history['phase1']['time'].append(t)
        print(f"[F1] Epoch {epoch+1:02d}/15 | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {t:.2f}s")

    # -------- FASE 2: Fine-tuning cuántico --------
    print("\n--- FASE 2 (fine-tuning cuántico) ---")
    model.unfreeze_quantum()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(6):
        start = time.time()
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        acc      = evaluate(model, val_loader, device)
        t        = time.time() - start

        history['phase2']['loss'].append(avg_loss)
        history['phase2']['acc'].append(acc)
        history['phase2']['time'].append(t)
        print(f"[F2] Epoch {epoch+1:02d}/6 | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {t:.2f}s")

    return history


# =========================
# VALIDACIÓN CRUZADA
# =========================
print("\nIniciando validación cruzada...")
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

all_acc, all_f1          = [], []
all_histories            = []
all_confusion_matrices   = []
class_names              = label_encoder.classes_

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor), 1):
    print(f"\n{'='*50}")
    print(f"FOLD {fold}/{KFOLDS}")
    print(f"{'='*50}")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(SEED)
    )

    model   = HybridModel()
    history = train_two_phase(model, train_loader, val_loader, fold, device)
    all_histories.append(history)

    acc = evaluate(model, val_loader, device)

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            p  = torch.argmax(model(xb), dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(yb.numpy())

    f1     = f1_score(labels, preds, average="macro")
    cm_fold = confusion_matrix(labels, preds)
    all_confusion_matrices.append(cm_fold)

    print(f"\nFold {fold} — Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"Matriz de confusión fold {fold}:\n{cm_fold}")

    all_acc.append(acc)
    all_f1.append(f1)

# =========================
# RESULTADOS FINALES
# =========================
print("\n" + "="*50)
print("RESULTADOS FINALES")
print("="*50)
print(f"Accuracy promedio : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"F1 promedio       : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

# =========================
# GRÁFICAS
# =========================
print("\nGenerando gráficas...")

# 1. Curvas de pérdida y accuracy por fold
fig, axes = plt.subplots(KFOLDS, 2, figsize=(14, 4 * KFOLDS))
fig.suptitle("Curvas de entrenamiento por fold", fontsize=14, fontweight="bold")

for fi, history in enumerate(all_histories):
    ax_loss, ax_acc = axes[fi]
    l1 = history['phase1']['loss']; l2 = history['phase2']['loss']
    a1 = history['phase1']['acc'];  a2 = history['phase2']['acc']
    ep1 = list(range(1, len(l1) + 1))
    ep2 = list(range(len(l1) + 1, len(l1) + len(l2) + 1))

    ax_loss.plot(ep1, l1, 'b-o', ms=4, label="Fase 1 (clásico)")
    ax_loss.plot(ep2, l2, 'r-s', ms=4, label="Fase 2 (cuántico)")
    ax_loss.axvline(len(l1) + .5, color='gray', ls='--', alpha=.5)
    ax_loss.set_title(f"Fold {fi+1} — Loss")
    ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=.3)

    ax_acc.plot(ep1, a1, 'b-o', ms=4, label="Fase 1 (clásico)")
    ax_acc.plot(ep2, a2, 'r-s', ms=4, label="Fase 2 (cuántico)")
    ax_acc.axvline(len(a1) + .5, color='gray', ls='--', alpha=.5)
    ax_acc.set_title(f"Fold {fi+1} — Accuracy")
    ax_acc.set_xlabel("Época"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {OUTPUT_DIR}/curvas_entrenamiento.png")

# 2. Matrices de confusión
fig, axes = plt.subplots(1, KFOLDS, figsize=(6 * KFOLDS, 5))
if KFOLDS == 1:
    axes = [axes]
fig.suptitle("Matrices de confusión por fold", fontsize=14, fontweight="bold")

for fi, cm in enumerate(all_confusion_matrices):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[fi])
    axes[fi].set_title(f"Fold {fi+1}")
    axes[fi].set_xlabel("Predicho"); axes[fi].set_ylabel("Real")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "matrices_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {OUTPUT_DIR}/matrices_confusion.png")

# 3. Resumen métricas por fold
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(KFOLDS); w = 0.35
b1 = ax.bar(x - w/2, all_acc, w, label="Accuracy", color="#4A3AE8", alpha=.85)
b2 = ax.bar(x + w/2, all_f1,  w, label="F1 macro",  color="#2E7D32", alpha=.85)
ax.axhline(np.mean(all_acc), color="#4A3AE8", ls='--', alpha=.5,
           label=f"Acc media {np.mean(all_acc):.3f}")
ax.axhline(np.mean(all_f1),  color="#2E7D32", ls='--', alpha=.5,
           label=f"F1 media  {np.mean(all_f1):.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(KFOLDS)])
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("Accuracy y F1 macro por fold")
ax.legend(); ax.grid(axis='y', alpha=.3)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + .01,
            f"{b.get_height():.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metricas_por_fold.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → {OUTPUT_DIR}/metricas_por_fold.png")

print("\nListo.")
