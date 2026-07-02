# ============================================
# CONFIGURACION DE LOGGING EN TIEMPO REAL
# ============================================
import sys
import os
import json
from datetime import datetime
import pickle

# FORZAR SALIDA EN TIEMPO REAL
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# Crear carpeta para logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Funcion de logging con timestamp y guardado en archivo
def log(msg, also_print=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    if also_print:
        print(log_msg, flush=True)
    
    # Guardar en archivo de log
    with open(LOG_DIR / "training_log.txt", "a") as f:
        f.write(log_msg + "\n")

# Guardar configuracion
def save_config():
    config = {
        "N_QUBITS": N_QUBITS,
        "N_LAYERS": N_LAYERS,
        "KFOLDS": KFOLDS,
        "EPOCHS_P1": EPOCHS_P1,
        "EPOCHS_P2": EPOCHS_P2,
        "LR_P1": LR_P1,
        "LR_P2": LR_P2,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_SAMPLES": MAX_SAMPLES,
        "SEED": SEED,
        "timestamp": datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log("Configuracion guardada")

# ============================================
# IMPORTS
# ============================================
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import copy
import warnings
warnings.filterwarnings('ignore')

import pennylane as qml

# ============================================
# CONFIGURACION MEJORADA
# ============================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CSV_PATH      = "mental_state.csv"
LABEL_COL     = "Label"

# HIPERPARAMETROS OPTIMIZADOS
N_QUBITS      = 14
N_LAYERS      = 4
N_CLASSES     = 3
N_WEIGHTS     = N_QUBITS * 3 * N_LAYERS

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
KFOLDS        = 3
MAX_SAMPLES   = 2400
BATCH_SIZE    = 64
EPOCHS_P1     = 35
EPOCHS_P2     = 20
PATIENCE_P1   = 10
LR_P1         = 1e-3
LR_P2         = 5e-4

OUTPUT_DIR = Path("resultados_v7")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(DEVICE)
log(f"Iniciando experimento v7")
log(f"Dispositivo PyTorch: {device}")
log(f"Output dir: {OUTPUT_DIR}")

# ============================================
# CARGA DE DATOS
# ============================================
log("Cargando datos...")
df = pd.read_csv(CSV_PATH)
log(f"Total muestras CSV: {len(df)}")

if MAX_SAMPLES is not None and MAX_SAMPLES < len(df):
    df, _ = train_test_split(df, train_size=MAX_SAMPLES,
                              stratify=df[LABEL_COL], random_state=SEED)
    log(f"Submuestreo: usando {MAX_SAMPLES} muestras")
else:
    log(f"Usando todos los datos: {len(df)} muestras")

X     = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

# ============================================
# PREPROCESAMIENTO
# ============================================
log("Preprocesando etiquetas...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
log(f"Clases: {label_encoder.classes_}")

log("Escalando datos...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

log(f"Aplicando PCA ({N_QUBITS} componentes)...")
pca = PCA(n_components=N_QUBITS, random_state=SEED)
X_reduced = pca.fit_transform(X_scaled)
log(f"Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.4f}")

# Guardar componentes PCA
np.save(OUTPUT_DIR / "pca_components.npy", pca.components_)
np.save(OUTPUT_DIR / "pca_mean.npy", pca.mean_)

log("Codificando angulos...")
min_val  = X_reduced.min(axis=0)
max_val  = X_reduced.max(axis=0)
X_angles = 2 * np.pi * (X_reduced - min_val) / (max_val - min_val + 1e-8) - np.pi

X_tensor = torch.tensor(X_angles, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

log(f"Dataset final: {X_tensor.shape}")
log(f"Distribucion de clases: {np.bincount(y)}")

# Guardar estadisticas
np.save(OUTPUT_DIR / "X_angles.npy", X_angles)
np.save(OUTPUT_DIR / "y_labels.npy", y)

save_config()

# ============================================
# CIRCUITO CUANTICO
# ============================================
log("Inicializando simulador cuantico...")
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    log("Simulador: lightning.qubit")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    log("Simulador: default.qubit (fallback)")

try:
    _test_dev = qml.device("lightning.qubit", wires=2)
    DIFF_METHOD = "adjoint"
    log("Metodo de diferenciacion: adjoint")
except Exception:
    DIFF_METHOD = "parameter-shift"
    log("Metodo de diferenciacion: parameter-shift")

@qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)

    idx = 0
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[idx],     wires=q)
            qml.RY(weights[idx + 1], wires=q)
            qml.RZ(weights[idx + 2], wires=q)
            idx += 3

        if layer % 2 == 0:
            for q in range(0, N_QUBITS - 1, 2):
                qml.CNOT(wires=[q, q + 1])
        else:
            for q in range(1, N_QUBITS - 1, 2):
                qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return tuple(qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS))

# ============================================
# MODELOS MEJORADOS
# ============================================
class QuantumLayer(nn.Module):
    def __init__(self, n_weights: int):
        super().__init__()
        init_w = torch.empty(n_weights).uniform_(-np.pi / 4, np.pi / 4)
        self.weights = nn.Parameter(init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.cpu()
        w_cpu = self.weights.cpu() if self.weights.is_cuda else self.weights
        rows = [
            torch.stack(list(quantum_circuit(x_cpu[i], w_cpu))).float()
            for i in range(x_cpu.shape[0])
        ]
        out = torch.stack(rows)
        return out.to(x.device)


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer(N_WEIGHTS)
        # Clasificador mejorado con mas capacidad
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(N_QUBITS),
            nn.Linear(N_QUBITS, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, N_CLASSES),
        )

    def freeze_quantum(self):
        self.quantum.weights.requires_grad = False

    def unfreeze_quantum(self):
        self.quantum.weights.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.quantum(x))


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        log_p = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_p, self.smoothing / (n_cls - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_dist * log_p).sum(dim=-1).mean()

# ============================================
# FUNCIONES DE ENTRENAMIENTO MEJORADAS
# ============================================
def evaluate(model: nn.Module, loader: DataLoader, dev: torch.device) -> tuple:
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            p  = torch.argmax(model(xb), dim=1)
            preds_all.extend(p.cpu().numpy())
            labels_all.extend(yb.numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")
    return acc, f1, labels_all, preds_all


def train_two_phase(model, train_loader, val_loader, fold, dev):
    criterion = LabelSmoothingCE(smoothing=0.1)
    history = {
        'phase1': {'loss': [], 'acc': [], 'f1': [], 'time': []},
        'phase2': {'loss': [], 'acc': [], 'f1': [], 'time': []},
    }

    log(f"\n{'='*60}")
    log(f"--- FASE 1 (clasico) Fold {fold} ---")
    log(f"{'='*60}")
    model.freeze_quantum()
    model = model.to(dev)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P1, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_acc, best_state, patience_c = 0.0, None, 0

    for epoch in range(EPOCHS_P1):
        t0 = time.time()
        model.train()
        ep_loss, n_batch = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1

        avg_loss = ep_loss / n_batch
        acc, f1, _, _ = evaluate(model, val_loader, dev)
        scheduler.step(acc)
        t = time.time() - t0

        history['phase1']['loss'].append(avg_loss)
        history['phase1']['acc'].append(acc)
        history['phase1']['f1'].append(f1)
        history['phase1']['time'].append(t)

        if acc > best_acc:
            best_acc, best_state, patience_c = acc, copy.deepcopy(model.state_dict()), 0
            # Guardar checkpoint del mejor modelo
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_phase1_fold{fold}.pt")
            log(f"  Checkpoint guardado (acc: {best_acc:.4f})")
        else:
            patience_c += 1

        cur_lr = optimizer.param_groups[0]['lr']
        log(f"[F1 Fold {fold}] Epoch {epoch+1:02d}/{EPOCHS_P1} | "
            f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
            f"Best: {best_acc:.4f} | LR: {cur_lr:.5f} | Time: {t:.1f}s")

        if patience_c >= PATIENCE_P1:
            log(f"  Early stop en epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"  Mejor acc fase 1: {best_acc:.4f}")

    log(f"\n{'='*60}")
    log(f"--- FASE 2 (fine-tuning cuantico) Fold {fold} ---")
    log(f"{'='*60}")
    model.unfreeze_quantum()

    optimizer = torch.optim.AdamW([
        {'params': model.quantum.parameters(),    'lr': LR_P2 * 0.3},
        {'params': model.classifier.parameters(), 'lr': LR_P2},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    best_acc_p2 = best_acc
    best_state  = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(EPOCHS_P2):
        t0 = time.time()
        model.train()
        ep_loss, n_batch = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batch += 1

        avg_loss = ep_loss / n_batch
        acc, f1, _, _ = evaluate(model, val_loader, dev)
        scheduler.step(acc)
        t = time.time() - t0

        history['phase2']['loss'].append(avg_loss)
        history['phase2']['acc'].append(acc)
        history['phase2']['f1'].append(f1)
        history['phase2']['time'].append(t)

        if acc > best_acc_p2:
            best_acc_p2 = acc
            best_state = copy.deepcopy(model.state_dict())
            best_f1 = f1
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_phase2_fold{fold}.pt")
            log(f"  Checkpoint guardado (acc: {best_acc_p2:.4f}, f1: {f1:.4f})")

        log(f"[F2 Fold {fold}] Epoch {epoch+1:02d}/{EPOCHS_P2} | "
            f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
            f"Best: {best_acc_p2:.4f} | Time: {t:.1f}s")

    model.load_state_dict(best_state)
    log(f"  Mejor acc fase 2: {best_acc_p2:.4f} (F1: {best_f1:.4f})")

    return history

# ============================================
# VALIDACION CRUZADA
# ============================================
log(f"\n{'='*60}")
log("INICIANDO VALIDACION CRUZADA")
log(f"{'='*60}")

skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

all_acc, all_f1 = [], []
all_histories = []
all_confusion_matrices = []
all_classification_reports = []
class_names = label_encoder.classes_

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor), 1):
    log(f"\n{'='*60}")
    log(f"FOLD {fold}/{KFOLDS}")
    log(f"{'='*60}")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

    log(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    pin = device.type == 'cuda'
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=pin,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=pin)

    model = HybridModel()
    log(f"Modelo creado. Parametros: {sum(p.numel() for p in model.parameters()):,}")

    history = train_two_phase(model, train_loader, val_loader, fold, device)
    all_histories.append(history)

    acc, f1, labels_list, preds = evaluate(model, val_loader, device)

    cm_fold = confusion_matrix(labels_list, preds)
    all_confusion_matrices.append(cm_fold)

    # Guardar classification report
    report = classification_report(labels_list, preds, target_names=class_names, output_dict=True)
    all_classification_reports.append(report)

    log(f"\n{'='*50}")
    log(f"RESULTADOS FOLD {fold}")
    log(f"{'='*50}")
    log(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    log(f"Matriz de confusion:\n{cm_fold}")

    # Mostrar precision por clase
    for cls in class_names:
        idx = list(class_names).index(cls)
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1_score_cls = report[cls]['f1-score']
        log(f"  {cls}: P={precision:.3f}, R={recall:.3f}, F1={f1_score_cls:.3f}")

    all_acc.append(acc)
    all_f1.append(f1)

total_time = time.time() - start_time

# ============================================
# RESULTADOS FINALES
# ============================================
log(f"\n{'='*60}")
log("RESULTADOS FINALES")
log(f"{'='*60}")
log(f"Accuracy promedio : {np.mean(all_acc):.4f} +/- {np.std(all_acc):.4f}")
log(f"F1 promedio       : {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")
log(f"Tiempo total      : {total_time/60:.2f} minutos")
log(f"Mejor accuracy    : {max(all_acc):.4f}")
log(f"Peor accuracy     : {min(all_acc):.4f}")

# Guardar resultados en JSON
results = {
    "accuracy": all_acc,
    "f1": all_f1,
    "mean_accuracy": np.mean(all_acc),
    "std_accuracy": np.std(all_acc),
    "mean_f1": np.mean(all_f1),
    "std_f1": np.std(all_f1),
    "best_accuracy": max(all_acc),
    "total_time_minutes": total_time/60,
    "config": {
        "N_QUBITS": N_QUBITS,
        "N_LAYERS": N_LAYERS,
        "EPOCHS_P1": EPOCHS_P1,
        "EPOCHS_P2": EPOCHS_P2,
        "LR_P1": LR_P1,
        "LR_P2": LR_P2,
        "BATCH_SIZE": BATCH_SIZE
    }
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================
# GRAFICAS
# ============================================
log("\nGenerando graficas...")

# 1. Curvas de entrenamiento
fig, axes = plt.subplots(KFOLDS, 2, figsize=(14, 4 * KFOLDS))
fig.suptitle(f"Curvas de entrenamiento por fold (v7)", fontsize=14, fontweight="bold")

for fi, history in enumerate(all_histories):
    ax_loss, ax_acc = axes[fi]
    l1 = history['phase1']['loss']; l2 = history['phase2']['loss']
    a1 = history['phase1']['acc'];  a2 = history['phase2']['acc']
    ep1 = list(range(1, len(l1) + 1))
    ep2 = list(range(len(l1) + 1, len(l1) + len(l2) + 1))

    ax_loss.plot(ep1, l1, 'b-o', ms=3, label="Fase 1")
    ax_loss.plot(ep2, l2, 'r-s', ms=3, label="Fase 2")
    ax_loss.axvline(len(l1) + .5, color='gray', ls='--', alpha=.5)
    ax_loss.set_title(f"Fold {fi+1} — Loss")
    ax_loss.set_xlabel("Epoca"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=.3)

    ax_acc.plot(ep1, a1, 'b-o', ms=3, label="Fase 1")
    ax_acc.plot(ep2, a2, 'r-s', ms=3, label="Fase 2")
    ax_acc.axvline(len(a1) + .5, color='gray', ls='--', alpha=.5)
    ax_acc.axhline(0.80, color='green', ls=':', alpha=.6, label="80% target")
    ax_acc.set_title(f"Fold {fi+1} — Accuracy")
    ax_acc.set_xlabel("Epoca"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Matrices de confusion
fig, axes = plt.subplots(1, KFOLDS, figsize=(6 * KFOLDS, 5))
if KFOLDS == 1:
    axes = [axes]
fig.suptitle("Matrices de confusion por fold", fontsize=14, fontweight="bold")
for fi, cm in enumerate(all_confusion_matrices):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[fi])
    axes[fi].set_title(f"Fold {fi+1} (Acc: {all_acc[fi]:.3f})")
    axes[fi].set_xlabel("Predicho"); axes[fi].set_ylabel("Real")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "matrices_confusion.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Metricas por fold
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(KFOLDS); w = 0.35
b1 = ax.bar(x - w/2, all_acc, w, label="Accuracy", color="#4A3AE8", alpha=.85)
b2 = ax.bar(x + w/2, all_f1,  w, label="F1 macro",  color="#2E7D32", alpha=.85)
ax.axhline(np.mean(all_acc), color="#4A3AE8", ls='--', alpha=.5,
           label=f"Acc media {np.mean(all_acc):.3f}")
ax.axhline(np.mean(all_f1),  color="#2E7D32", ls='--', alpha=.5,
           label=f"F1 media  {np.mean(all_f1):.3f}")
ax.axhline(0.80, color='red', ls=':', alpha=.7, label="80% target")
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

log(f"\nExperimento completado. Resultados en: {OUTPUT_DIR}")
log("Archivos generados:")
for f in OUTPUT_DIR.iterdir():
    log(f"  - {f.name}")

log("Checkpoints guardados:")
for f in CHECKPOINT_DIR.iterdir():
    log(f"  - {f.name}")
