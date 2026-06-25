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
import warnings
warnings.filterwarnings('ignore')

import pennylane as qml

# =========================
# FIJAR SEMILLAS
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Mejor para rendimiento

# =========================
# CONFIGURACIÓN
# =========================
CSV_PATH   = "mental_state.csv"
LABEL_COL  = "Label"

N_QUBITS   = 12
N_LAYERS   = 3
N_CLASSES  = 3
N_WEIGHTS  = N_QUBITS * 3 * N_LAYERS

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
KFOLDS      = 3
MAX_SAMPLES = 2400
BATCH_SIZE  = 64
EPOCHS_PHASE1 = 20  # Aumentado
EPOCHS_PHASE2 = 10  # Aumentado

OUTPUT_DIR = Path("resultados_entrenamiento_optimizado")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(DEVICE)
print(f"Dispositivo PyTorch: {device}")

# =========================
# CARGA Y PREPARACIÓN
# =========================
print("Cargando datos...")
df = pd.read_csv(CSV_PATH)
print(f"Total de muestras en CSV: {len(df)}")

if MAX_SAMPLES is not None and MAX_SAMPLES < len(df):
    df, _ = train_test_split(
        df,
        train_size=MAX_SAMPLES,
        stratify=df[LABEL_COL],
        random_state=SEED
    )
    print(f"Submuestreo aplicado: usando {MAX_SAMPLES} muestras")
else:
    print(f"Usando todos los datos disponibles: {len(df)} muestras")

X     = df.drop(columns=[LABEL_COL]).values.astype(np.float32)
y_raw = df[LABEL_COL].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_QUBITS)
X_reduced = pca.fit_transform(X_scaled)

min_val = X_reduced.min(axis=0)
max_val = X_reduced.max(axis=0)
X_angles = 2 * np.pi * (X_reduced - min_val) / (max_val - min_val + 1e-8) - np.pi

X_tensor = torch.tensor(X_angles, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"Dataset final: {X_tensor.shape}")
print(f"Distribución de clases: {np.bincount(y)}")
print(f"Varianza explicada por PCA: {pca.explained_variance_ratio_.sum():.4f}")

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
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor):
    # Data encoding con más rotaciones
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    
    # Capas variacionales
    idx = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[idx], wires=q)
            qml.RY(weights[idx + 1], wires=q)
            qml.RZ(weights[idx + 2], wires=q)
            idx += 3
        
        # Entanglement con conexiones alternadas
        for q in range(0, N_QUBITS - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        for q in range(1, N_QUBITS - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS))

# =========================
# CAPA CUÁNTICA
# =========================
class QuantumLayer(nn.Module):
    def __init__(self, n_weights: int):
        super().__init__()
        init_w = torch.empty(n_weights).uniform_(-np.pi/4, np.pi/4)
        self.weights = nn.Parameter(init_w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        weights_np = self.weights.detach().cpu().numpy()
        
        results = []
        for i in range(batch_size):
            result = quantum_circuit(x_np[i], weights_np)
            results.append(result)
        
        return torch.tensor(np.array(results), dtype=torch.float32, device=x.device)

# =========================
# MODELO HÍBRIDO MEJORADO
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer(N_WEIGHTS)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(N_QUBITS),
            nn.Linear(N_QUBITS, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, N_CLASSES)
        )
    
    def freeze_quantum(self):
        self.quantum.weights.requires_grad = False
    
    def unfreeze_quantum(self):
        self.quantum.weights.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.quantum(x))

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
# ENTRENAMIENTO MEJORADO
# =========================
def train_two_phase(model, train_loader, val_loader, fold, device):
    criterion = nn.CrossEntropyLoss()
    history = {
        'phase1': {'loss': [], 'acc': [], 'time': []},
        'phase2': {'loss': [], 'acc': [], 'time': []}
    }
    
    # -------- FASE 1 --------
    print("\n--- FASE 1 (entrenamiento clásico) ---")
    model.freeze_quantum()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS_PHASE1):
        start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        acc = evaluate(model, val_loader, device)
        scheduler.step(acc)
        t = time.time() - start
        
        history['phase1']['loss'].append(avg_loss)
        history['phase1']['acc'].append(acc)
        history['phase1']['time'].append(t)
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_model_fold{fold}_phase1.pt")
        else:
            patience_counter += 1
        
        print(f"[F1] Epoch {epoch+1:02d}/{EPOCHS_PHASE1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t:.2f}s")
        
        if patience_counter >= 5:
            print(f"Early stopping en epoch {epoch+1}")
            break
    
    # Cargar mejor modelo de fase 1
    model.load_state_dict(torch.load(OUTPUT_DIR / f"best_model_fold{fold}_phase1.pt"))
    
    # -------- FASE 2 --------
    print("\n--- FASE 2 (fine-tuning cuántico) ---")
    model.unfreeze_quantum()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    for epoch in range(EPOCHS_PHASE2):
        start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        acc = evaluate(model, val_loader, device)
        scheduler.step(acc)
        t = time.time() - start
        
        history['phase2']['loss'].append(avg_loss)
        history['phase2']['acc'].append(acc)
        history['phase2']['time'].append(t)
        
        print(f"[F2] Epoch {epoch+1:02d}/{EPOCHS_PHASE2} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {t:.2f}s")
    
    return history

# =========================
# VALIDACIÓN CRUZADA
# =========================
print("\nIniciando validación cruzada...")
skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

all_acc, all_f1 = [], []
all_histories = []
all_confusion_matrices = []
class_names = label_encoder.classes_

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
        pin_memory=True if device.type == 'cuda' else False,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    model = HybridModel()
    history = train_two_phase(model, train_loader, val_loader, fold, device)
    all_histories.append(history)
    
    # Evaluación final
    acc = evaluate(model, val_loader, device)
    
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            p = torch.argmax(model(xb), dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(yb.numpy())
    
    f1 = f1_score(labels, preds, average="macro")
    cm_fold = confusion_matrix(labels, preds)
    all_confusion_matrices.append(cm_fold)
    
    print(f"\nFold {fold} — Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"Matriz de confusión:\n{cm_fold}")
    
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

# 1. Curvas de entrenamiento
fig, axes = plt.subplots(KFOLDS, 2, figsize=(14, 4 * KFOLDS))
fig.suptitle("Curvas de entrenamiento por fold", fontsize=14, fontweight="bold")

for fi, history in enumerate(all_histories):
    ax_loss, ax_acc = axes[fi]
    l1 = history['phase1']['loss']; l2 = history['phase2']['loss']
    a1 = history['phase1']['acc'];  a2 = history['phase2']['acc']
    ep1 = list(range(1, len(l1) + 1))
    ep2 = list(range(len(l1) + 1, len(l1) + len(l2) + 1))
    
    ax_loss.plot(ep1, l1, 'b-o', ms=4, label="Fase 1")
    ax_loss.plot(ep2, l2, 'r-s', ms=4, label="Fase 2")
    ax_loss.axvline(len(l1) + .5, color='gray', ls='--', alpha=.5)
    ax_loss.set_title(f"Fold {fi+1} — Loss")
    ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(alpha=.3)
    
    ax_acc.plot(ep1, a1, 'b-o', ms=4, label="Fase 1")
    ax_acc.plot(ep2, a2, 'r-s', ms=4, label="Fase 2")
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

# 3. Resumen métricas
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(KFOLDS); w = 0.35
b1 = ax.bar(x - w/2, all_acc, w, label="Accuracy", color="#4A3AE8", alpha=.85)
b2 = ax.bar(x + w/2, all_f1,  w, label="F1 macro", color="#2E7D32", alpha=.85)
ax.axhline(np.mean(all_acc), color="#4A3AE8", ls='--', alpha=.5,
           label=f"Acc media {np.mean(all_acc):.3f}")
ax.axhline(np.mean(all_f1), color="#2E7D32", ls='--', alpha=.5,
           label=f"F1 media {np.mean(all_f1):.3f}")
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

print("\n✅ Entrenamiento completado!")
print(f"Resultados guardados en: {OUTPUT_DIR}")
