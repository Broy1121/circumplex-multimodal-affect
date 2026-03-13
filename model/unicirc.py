import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="Mean of empty slice")
import json

# ═══════════════════════════════════════════════════════════
# PATH RESOLUTION
# Always resolve data paths relative to this file's location,
# not the current working directory.
#
# Project layout:
#   circumplex-multimodal-affect/
#   ├── model/
#   │   └── unicirc.py          ← __file__  (this script)
#   └── data/
#       └── processed_mosei.pkl
#
# Works correctly whether run as:
#   python model/unicirc.py      (from project root)
#   python unicirc.py            (from inside model/)
#   !python model/unicirc.py     (from Colab)
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR   = Path(__file__).resolve().parent   # → .../model/
PROJECT_ROOT = SCRIPT_DIR.parent                 # → .../circumplex-multimodal-affect/
DATA_DIR     = PROJECT_ROOT / "data"             # → .../data/

DEFAULT_MOSEI_PATH = DATA_DIR / "processed_mosei.pkl"


# ═══════════════════════════════════════════════════════════
# DEVICE SETUP
# Automatically uses GPU if available, otherwise CPU.
# GPU makes training ~10x faster than CPU.
# ═══════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# Fixed sequence lengths for each modality.
# Shorter sequences → padded with zeros to reach these lengths.
# Longer sequences  → truncated to these lengths.
#
#   MAX_TEXT   = 40   words per utterance  (WORDVEC, 300-dim GloVe)
#   MAX_AUDIO  = 600  frames per utterance (COVAREP, 74-dim acoustic)
#   MAX_VISION = 200  frames per utterance (FACET,   35-dim facial)
# ═══════════════════════════════════════════════════════════
MAX_TEXT   = 40
MAX_AUDIO  = 600
MAX_VISION = 200


# ═══════════════════════════════════════════════════════════
# SECTION 2 — HELPER FUNCTIONS
# Small utility functions used during data loading.
# ═══════════════════════════════════════════════════════════

def extract_va(label_features):
    """
    Extracts Valence and Arousal from MOSEI's 7-dim label vector.

    MOSEI label format:
        [0] sentiment score → used as Valence  (range ~-3 to +3)
        [1] happiness   ┐
        [3] anger       ├── max of these 4 = Arousal proxy
        [4] surprise    │   (high-activation emotions)
        [6] fear        ┘

    Why this approach:
        MOSEI has no direct arousal annotation.
        High-energy emotions (happiness, anger, surprise, fear)
        correlate with high arousal in Russell's circumplex model.
    """
    valence = float(label_features[0])
    arousal = float(np.max(label_features[[1, 3, 4, 6]]))
    return valence, arousal


def clean_features(features):
    """
    Replaces Inf and -Inf values with 0.0.

    Why needed:
        COVAREP audio features in MOSEI contain Inf values
        caused by division-by-zero during feature extraction.
        Inf values propagate through the network and cause NaN loss.
        Replacing with 0 is safe — it means 'no signal' for that frame.
    """
    features = features.copy()               # avoid mutating the original array
    features[~np.isfinite(features)] = 0.0
    return features


def pad_sequence_2d(features, max_len):
    """
    Pads or truncates a (T, D) feature matrix to (max_len, D).

    Why needed:
        Each utterance has a different number of frames/words.
        Neural networks require fixed-size inputs in a batch.
        Short sequences → pad with zeros at the end.
        Long sequences  → truncate to max_len.

    Args:
        features: numpy array of shape (T, D)
        max_len:  target sequence length
    Returns:
        numpy array of shape (max_len, D)
    """
    T, D = features.shape
    if T >= max_len:
        return features[:max_len]
    pad = np.zeros((max_len - T, D), dtype=np.float32)
    return np.vstack([features, pad])


# ═══════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# Loads processed_mosei.pkl from disk.
# The pickle contains pre-extracted features for ~10,913
# utterances from CMU-MOSEI YouTube video clips.
# ═══════════════════════════════════════════════════════════

def load_mosei(path=None):
    """
    Loads the processed MOSEI pickle file.

    Args:
        path: relative path to processed_mosei.pkl
    Returns:
        mosei dict {video_id: [utterance_dicts]}
    """
    # Use DEFAULT_MOSEI_PATH if no path given — resolves relative to this file
    if path is None:
        path = DEFAULT_MOSEI_PATH
    path = Path(path)
    print(f"Loading MOSEI from {path} ...")
    with open(path, "rb") as f:
        mosei = pickle.load(f)
    print(f"Loaded {len(mosei)} videos")
    return mosei


# ═══════════════════════════════════════════════════════════
# SECTION 4 — DATASET CLASS
# Converts raw MOSEI pickle into PyTorch tensors.
# Each sample: text (40,300) + audio (600,74) + vision (200,35)
#              + valence (scalar) + arousal (scalar)
# ═══════════════════════════════════════════════════════════

class MOSEITemporalDataset(Dataset):
    """
    PyTorch Dataset for CMU-MOSEI multimodal emotion data.

    Structure of processed_mosei.pkl:
        dict {
            video_id: [
                {
                    'WORDVEC': { 'features': (T, 300) },  # GloVe text
                    'COVAREP': { 'features': (T, 74)  },  # acoustic
                    'FACET':   { 'features': (T, 35)  },  # facial
                    'Labels':  { 'features': (7,)     },  # emotion labels
                },
                ...  # one dict per utterance segment
            ]
        }
    """
    def __init__(self, mosei_dict):
        self.samples = []

        for video_id, utterances in mosei_dict.items():
            for seg in utterances:
                try:
                    # Unwrap numpy object scalars (.item() extracts the dict)
                    labels_raw  = np.array(seg['Labels']).item()
                    covarep_raw = np.array(seg['COVAREP']).item()
                    facet_raw   = np.array(seg['FACET']).item()
                    wordvec_raw = np.array(seg['WORDVEC']).item()

                    # Clean Inf values then pad/truncate to fixed lengths
                    text   = pad_sequence_2d(
                        clean_features(wordvec_raw['features'].astype(np.float32)),
                        MAX_TEXT)      # → (40,  300)

                    audio  = pad_sequence_2d(
                        clean_features(covarep_raw['features'].astype(np.float32)),
                        MAX_AUDIO)     # → (600, 74)

                    vision = pad_sequence_2d(
                        clean_features(facet_raw['features'].astype(np.float32)),
                        MAX_VISION)    # → (200, 35)

                    # Extract VA labels from 7-dim label vector
                    valence, arousal = extract_va(labels_raw['features'])

                    # Skip samples with invalid labels
                    if not (np.isfinite(valence) and np.isfinite(arousal)):
                        continue

                    self.samples.append({
                        'text':    torch.tensor(text),
                        'audio':   torch.tensor(audio),
                        'vision':  torch.tensor(vision),
                        'valence': torch.tensor(valence, dtype=torch.float32),
                        'arousal': torch.tensor(arousal, dtype=torch.float32),
                    })

                except Exception as e:
                    print(f"Skipped {video_id}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Stacks a list of sample dicts into a single batched dict.
    Called automatically by DataLoader for each batch.

    Input:  list of 32 individual sample dicts
    Output: single dict with tensors of shape (32, T, D)
    """
    return {
        'text':    torch.stack([s['text']    for s in batch]),  # (B, 40,  300)
        'audio':   torch.stack([s['audio']   for s in batch]),  # (B, 600, 74)
        'vision':  torch.stack([s['vision']  for s in batch]),  # (B, 200, 35)
        'valence': torch.stack([s['valence'] for s in batch]),  # (B,)
        'arousal': torch.stack([s['arousal'] for s in batch]),  # (B,)
    }


# ═══════════════════════════════════════════════════════════
# Now loads the pickle automatically when run as a script.
# Also available when imported as a module in a notebook:
#   from model.unicirc import MultimodalTemporalModel, dataset
# ═══════════════════════════════════════════════════════════
mosei   = load_mosei()   # uses DEFAULT_MOSEI_PATH automatically
dataset = MOSEITemporalDataset(mosei)
print(f"Total utterances: {len(dataset)}")


# ═══════════════════════════════════════════════════════════
# SECTION 5 — MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """
    Encodes a variable-length sequence into a fixed-size vector.
    Used independently for each modality (text, audio, vision).

    Architecture:
        Input (B, T, D)
            ↓
        Bidirectional GRU     → reads sequence forward and backward
            ↓
        Attention pooling     → weights each timestep by importance
            ↓
        Linear projection     → compress to hidden_dim
            ↓
        LayerNorm             → stabilise activations
            ↓
        Output (B, hidden_dim)

    Why Bidirectional GRU:
        Forward pass  → understands context from past words/frames
        Backward pass → understands context from future words/frames
        Together      → full temporal context for each position

    Why Attention pooling instead of mean pooling:
        Not all frames are equally important emotionally.
        A peak moment of anger matters more than neutral frames.
        Attention learns to weight important moments higher.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()

        # Bidirectional GRU — output size is hidden_dim * 2 (forward + backward)
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # input shape: (batch, seq, features)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention: assigns a scalar importance weight to each timestep
        self.attn = nn.Linear(hidden_dim * 2, 1)

        # Project from hidden_dim*2 → hidden_dim
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # LayerNorm stabilises training
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        out, _  = self.gru(x)                           # (B, T, H*2)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1) — sums to 1
        pooled  = (out * weights).sum(dim=1)            # (B, H*2) — weighted average
        return self.norm(self.proj(pooled))             # (B, H)


class MultimodalTemporalModel(nn.Module):
    """
    Full multimodal emotion model — UniCirc.

    Architecture:
        Text   (B, 40,  300) → TemporalEncoder → (B, 128) ─┐
        Audio  (B, 600, 74)  → TemporalEncoder → (B, 128) ──┼→ Transformer → (B, 384) → [V, A]
        Vision (B, 200, 35)  → TemporalEncoder → (B, 128) ─┘

    Why Transformer for fusion:
        Self-attention over 3 modality tokens lets each modality
        attend to the others. Text can influence audio interpretation
        and vice versa — capturing cross-modal emotion cues.

    Output:
        valence: (B,) — positive/negative emotion  (range ~-3 to +3)
        arousal: (B,) — calm/excited activation    (range  0 to +3)
    """
    def __init__(self, hidden_dim=128, n_heads=4, dropout=0.3):
        super().__init__()

        # Three separate encoders — one per modality
        # Input dims match the feature sizes in MOSEI:
        #   text:   300-dim GloVe word vectors
        #   audio:  74-dim  COVAREP acoustic features
        #   vision: 35-dim  FACET facial action unit features
        self.text_enc   = TemporalEncoder(300, hidden_dim, dropout=dropout)
        self.audio_enc  = TemporalEncoder(74,  hidden_dim, dropout=dropout)
        self.vision_enc = TemporalEncoder(35,  hidden_dim, dropout=dropout)

        # Transformer fusion over 3 modality tokens
        # d_model=128  → each token is 128-dim
        # nhead=4      → 4 attention heads, each focuses on different relationships
        # num_layers=2 → two rounds of cross-modal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # VA prediction head
        # Input:  3 fused modality vectors concatenated → 128*3 = 384
        # Output: 2 numbers → [valence, arousal]
        self.va_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, text, audio, vision):
        # Encode each modality independently → (B, 1, H)
        t = self.text_enc(text).unsqueeze(1)      # (B, 1, 128)
        a = self.audio_enc(audio).unsqueeze(1)    # (B, 1, 128)
        v = self.vision_enc(vision).unsqueeze(1)  # (B, 1, 128)

        # Stack as 3 tokens for Transformer → (B, 3, 128)
        tokens = torch.cat([t, a, v], dim=1)

        # Cross-modal attention fusion → (B, 3, 128)
        fused = self.fusion(tokens)

        # Flatten all 3 fused vectors → (B, 384)
        flat = fused.reshape(fused.size(0), -1)

        # Predict VA → (B, 2)
        out = self.va_head(flat)

        return out[:, 0], out[:, 1]   # valence, arousal


# ═══════════════════════════════════════════════════════════
# SECTION 6 — LOSS FUNCTION
# Combined MSE + CCC loss for VA regression.
# ═══════════════════════════════════════════════════════════

def concordance_correlation_coefficient(pred, target):
    """
    Concordance Correlation Coefficient (CCC).
    Standard metric for VA prediction in emotion recognition.

    Range: -1 (perfect inverse) to +1 (perfect agreement)
    > 0.5 = good,  > 0.6 = competitive with literature

    Why CCC instead of just MSE:
        MSE only measures numerical distance.
        CCC also measures whether predictions track the
        correct direction and scale of emotion changes.
        A model that always predicts the mean gets low MSE
        but CCC=0 — CCC catches this failure mode.

    Fix vs original:
        Added 1e-8 to numerator to prevent NaN when
        both pred and target have zero variance.
    """
    pred_mean,  target_mean = pred.mean(),  target.mean()
    pred_var,   target_var  = pred.var(),   target.var()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()

    ccc = (2 * covariance + 1e-8) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    )
    return ccc


def va_loss(v_pred, a_pred, v_true, a_true, alpha=0.5):
    """
    Combined loss = alpha * MSE + (1 - alpha) * (1 - CCC)

    alpha=0.5 weights MSE and CCC equally.

    MSE component:  penalises large numerical errors
    CCC component:  penalises wrong direction/scale predictions
    Together:       model must be both numerically accurate
                    and directionally correct
    """
    mse_v = nn.functional.mse_loss(v_pred, v_true)
    mse_a = nn.functional.mse_loss(a_pred, a_true)
    ccc_v = 1 - concordance_correlation_coefficient(v_pred, v_true)
    ccc_a = 1 - concordance_correlation_coefficient(a_pred, a_true)
    return alpha * (mse_v + mse_a) + (1 - alpha) * (ccc_v + ccc_a)


# ═══════════════════════════════════════════════════════════
# SECTION 7 — EPOCH RUNNER
# Runs one full pass through a dataloader (train or eval).
# Returns average loss, CCC-V, CCC-A for that pass.
#
# Fix vs original:
#   model is now passed as an argument instead of relying
#   on a global variable — safer and reusable.
# ═══════════════════════════════════════════════════════════

def run_epoch(model, loader, train=True):
    """
    Runs one epoch (full pass through the dataset).

    If train=True:
        Model weights are updated via backpropagation.
        Gradients are computed and clipped.
    If train=False:
        No weight updates (validation/test mode).
        Faster, uses less memory.

    Returns:
        avg_loss, avg_ccc_v, avg_ccc_a  for this epoch
    """
    model.train() if train else model.eval()
    total_loss, ccc_v_sum, ccc_a_sum = 0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            text   = batch['text'].to(device)
            audio  = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            v_true = batch['valence'].to(device)
            a_true = batch['arousal'].to(device)

            # Forward pass — predict VA
            v_pred, a_pred = model(text, audio, vision)

            # Compute combined loss
            loss = va_loss(v_pred, a_pred, v_true, a_true)

            if train:
                optimizer.zero_grad()                                     # clear old gradients
                loss.backward()                                           # compute new gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip to prevent explosion
                optimizer.step()                                          # update weights

            total_loss += loss.item()
            ccc_v_sum  += concordance_correlation_coefficient(v_pred.detach(), v_true).item()
            ccc_a_sum  += concordance_correlation_coefficient(a_pred.detach(), a_true).item()

    n = len(loader)
    return total_loss / n, ccc_v_sum / n, ccc_a_sum / n


# ═══════════════════════════════════════════════════════════
# MAIN — runs when executed directly:  python model/unicirc.py
#
# When imported as a module in a notebook:
#   from model.unicirc import MultimodalTemporalModel
# all classes and functions above load, but the training
# loop below does NOT run automatically.
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Confirm sample shapes after loading ───────────────
    s = dataset[0]
    print(f"text={s['text'].shape} audio={s['audio'].shape} vision={s['vision'].shape}")
    print(f"valence={s['valence'].item():.4f}  arousal={s['arousal'].item():.4f}")

    # ── Build dataloaders ─────────────────────────────────
    # Split 80/10/10 → train / val / test
    # Fixed seed=42 ensures the same split every run (reproducibility)
    total   = len(dataset)
    n_train = int(0.8 * total)
    n_val   = int(0.1 * total)
    n_test  = total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # batch_size=32 balances GPU memory and training stability
    # shuffle=True on train → prevents model learning data order
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")

    # ── Model initialisation ──────────────────────────────
    model = MultimodalTemporalModel(hidden_dim=128).to(device)

    # AdamW: lr=3e-4 (step size), weight_decay=1e-4 (L2 regularisation)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Cosine schedule: large lr early (fast learning) → tiny lr late (fine-tuning)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # ── Sanity check ──────────────────────────────────────
    # One forward pass before training to confirm shapes and no NaNs
    batch = next(iter(train_loader))
    with torch.no_grad():
        vp, ap = model(
            batch['text'].to(device),
            batch['audio'].to(device),
            batch['vision'].to(device)
        )

    print(f"\nForward pass OK — v_pred:{vp.shape}  a_pred:{ap.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"NaNs in output: {torch.isnan(vp).any() or torch.isnan(ap).any()}")

    # ── Training loop ─────────────────────────────────────
    # Trains up to 30 epochs with early stopping.
    # Best checkpoint saved automatically to best_temporal_model.pt
    #
    # Early stopping:
    #   Stops if val loss doesn't improve for PATIENCE consecutive
    #   epochs — avoids wasted compute and overfitting.
    #   Previous runs showed best at epoch 15, so PATIENCE=7
    #   would have stopped at epoch 22, saving ~8 epochs of compute.

    EPOCHS        = 30
    PATIENCE      = 7
    best_val_loss = float('inf')
    patience_ctr  = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):

        # Training pass — weights update on every batch
        tr_loss, tr_ccc_v, tr_ccc_a = run_epoch(model, train_loader, train=True)

        # Validation pass — no weight updates, honest performance check
        vl_loss, vl_ccc_v, vl_ccc_a = run_epoch(model, val_loader,   train=False)

        # Reduce learning rate according to cosine schedule
        scheduler.step()

        # Record metrics for later plotting
        history.append({
            'epoch':    epoch,
            'tr_loss':  tr_loss,  'vl_loss':  vl_loss,
            'tr_ccc_v': tr_ccc_v, 'vl_ccc_v': vl_ccc_v,
            'tr_ccc_a': tr_ccc_a, 'vl_ccc_a': vl_ccc_a,
        })

        # Save checkpoint if best val loss so far
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), "best_temporal_model.pt")
            tag = " ← best"
        else:
            patience_ctr += 1
            tag = ""

        # Print progress on epoch 1 and every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Ep {epoch:02d} | "
                f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"CCC-V {tr_ccc_v:.3f}/{vl_ccc_v:.3f} | "
                f"CCC-A {tr_ccc_a:.3f}/{vl_ccc_a:.3f}{tag}"
            )

        # Early stopping check
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — "
                  f"no improvement for {PATIENCE} epochs")
            break

    with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)
    print("Training history saved → training_history.json")
    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Best checkpoint saved → best_temporal_model.pt")
