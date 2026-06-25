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
from pennylane import numpy as pnp

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
CSV_PATH = "mental_state.csv"
LABEL_COL = "Label"

N_QUBITS = 10
N_LAYERS = 2
N_CLASSES = 3

# Detectar GPU para PennyLane
device_name = "default.qubit"  # CPU por defecto
if torch.cuda.is_available():
    # PennyLane no tiene soporte nativo CUDA, pero podemos usar
    # lightning.qubit con GPU si está disponible
    try:
        import pennylane_lightning as lightning
        device_name = "lightning.qubit"
        print("Usando lightning.qubit (GPU/CPU optimizado)")
    except ImportError:
        device_name = "default.qubit"
        print("Usando default.qubit (CPU)")
else:
    print("Usando default.qubit (CPU)")

KFOLDS = 3
MAX_SAMPLES = 2000
BATCH_SIZE = 32

OUTPUT_DIR = Path("resultados_entrenamiento_pennylane")
OUTPUT_DIR.mkdir(exist_ok=True)

device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device_torch}")

# =========================
# CARGA Y PREPARACIÓN
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

# Escalar a [-pi, pi]
min_val = X_reduced.min(axis=0)
max_val = X_reduced.max(axis=0)
X_angles = 2 * np.pi * (X_reduced - min_val) / (max_val - min_val + 1e-8) - np.pi

X_tensor = torch.tensor(X_angles, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print(f"Dataset final: {X_tensor.shape}")
print(f"Distribución de clases: {np.bincount(y)}")

# =========================
# CIRCUITO CUÁNTICO CON PENNYLANE
# =========================
def create_vqc(n_qubits, n_layers):
    """Crea un circuito cuántico variacional con PennyLane"""
    
    # Definir los parámetros
    n_params = n_qubits * 3 * n_layers
    
    # Crear el device
    dev = qml.device(device_name, wires=n_qubits, shots=None)
    
    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        # Data encoding
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
            qml.RY(inputs[i], wires=i)
        
        # Capas variacionales
        idx = 0
        for _ in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[idx], wires=i)
                qml.RY(weights[idx + 1], wires=i)
                qml.RZ(weights[idx + 2], wires=i)
                idx += 3
            
            # Entanglement ring
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        
        # Medir todos los qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit, n_params

print("Creando circuito cuántico con PennyLane...")
quantum_circuit, n_weights = create_vqc(N_QUBITS, N_LAYERS)
print(f"Número de parámetros cuánticos: {n_weights}")

# =========================
# CAPA CUÁNTICA PARA PYTORCH
# =========================
class QuantumLayer(nn.Module):
    """Capa cuántica de PennyLane para PyTorch"""
    def __init__(self, circuit, n_weights):
        super().__init__()
        self.circuit = circuit
        
        # Inicializar pesos cuánticos
        init_weights = 0.01 * np.random.randn(n_weights)
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
    
    def forward(self, x):
        """Forward pass con batch processing"""
        batch_size = x.shape[0]
        
        # Procesar cada muestra del batch
        outputs = []
        for i in range(batch_size):
            # Convertir input a numpy para PennyLane
            inputs = x[i].detach().cpu().numpy()
            weights = self.weights.detach().cpu().numpy()
            
            # Ejecutar el circuito cuántico
            result = self.circuit(inputs, weights)
            outputs.append(result)
        
        # Convertir a tensor de PyTorch y mover al dispositivo
        return torch.tensor(np.array(outputs), dtype=torch.float32, device=x.device)

# =========================
# MODELO HÍBRIDO
# =========================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer(quantum_circuit, n_weights)
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
        for param in self.quantum.parameters():
            param.requires_grad = False
    
    def unfreeze_quantum(self):
        for param in self.quantum.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        q_out = self.quantum(x)
        return self.classifier(q_out)

# =========================
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# =========================
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

def train_two_phase(model, train_loader, val_loader, fold, device):
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'phase1': {'loss': [], 'acc': [], 'time': []},
        'phase2': {'loss': [], 'acc': [], 'time': []}
    }
    
    # -------- FASE 1 (Clásica) --------
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
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        acc = evaluate(model, val_loader, device)
        t = time.time() - start
        
        history['phase1']['loss'].append(avg_loss)
        history['phase1']['acc'].append(acc)
        history['phase1']['time'].append(t)
        
        print(f"[F1] Epoch {epoch+1:02d}/15 | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Time: {t:.2f}s")
    
    # -------- FASE 2 (Fine-tuning cuántico) --------
    print("\n--- FASE 2 (fine-tuning cuántico) ---")
    model.unfreeze_quantum()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )
    
    for epoch in range(6):
        start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        acc = evaluate(model, val_loader, device)
        t = time.time() - start
        
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

all_acc = []
all_f1 = []
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
        generator=torch.Generator().manual_seed(SEED)
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    model = HybridModel()
    
    history = train_two_phase(model, train_loader, val_loader, fold, device_torch)
    all_histories.append(history)
    
    acc = evaluate(model, val_loader, device_torch)
    
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device_torch)
            p = torch.argmax(model(xb), dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(yb.numpy())
    
    f1 = f1_score(labels, preds, average="macro")
    cm_fold = confusion_matrix(labels, preds)
    all_confusion_matrices.append(cm_fold)
    
    print(f"\nFold {fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Matriz de confusión fold {fold}:")
    print(cm_fold)
    
    all_acc.append(acc)
    all_f1.append(f1)

print("\n" + "="*50)
print("RESULTADOS FINALES")
print("="*50)
print(f"Accuracy promedio: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"F1 promedio: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
