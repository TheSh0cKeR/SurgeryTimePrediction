#!/usr/bin/env python
# best_nn_surgery_time_v4.py
# Deep Tabular NN  +  five-fold Optuna  +  15 % hold-out final test
# ---------------------------------------------------------------

import os, json, math, warnings, random, optuna, collections
from datetime import datetime
import numpy as np
import pandas as pd

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# ╭────────────────────────── CONFIG ───────────────────────────╮
DATA_PATH   = "Cleaned_Dataset_14minPlus.csv"
TARGET      = "Total Surgery Time"
LOG_TARGET  = False

N_SPLITS    = 5           # CV folds for Optuna
MAX_EPOCHS  = 200
PATIENCE    = 15
N_TRIALS    = 25
SEED        = 42

ID_THRESH   = 0.90        # drop columns with >90 % unique values (IDs)
SMALL_CARD  = 15          # small int → treat as categorical
KEEP_TOP_K  = 200         # cap rare string levels
MULTIVALUE_COLS = [
    "Cleaned Drug Names", "Cleaned Allergy Drug Names",
    "Semi Cleaned Drug Names", "Disease ICD9 Codes"
]
PROC_COL    = "Procedure Code"
TOP_PROC_K  = 40          # one-hot for the 40 most common codes
# ╰─────────────────────────────────────────────────────────────╯

torch.manual_seed(SEED);  np.random.seed(SEED);  random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Using device:", device)

# ╭─────────────────── 0. LOAD & 15 % TEST SPLIT ───────────────╮
df_full = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])

# stratify by quartile of target
strat = pd.qcut(df_full[TARGET], 4, labels=False, duplicates="drop")
train_df, test_df = train_test_split(
    df_full, test_size=0.15, random_state=SEED, stratify=strat
)
print(f"Train rows: {len(train_df)}  •  Test rows: {len(test_df)}")
df = train_df.copy()                                # used for all fitting
# ╰─────────────────────────────────────────────────────────────╯

# ╭──────────────────── 1. PRE-PROCESSING (fit on train) ───────╮
# 1-A  drop near-identifier cols
id_cols = [c for c in df.columns if c != TARGET and df[c].nunique() / len(df) > ID_THRESH]
df.drop(columns=id_cols, inplace=True)
test_df.drop(columns=[c for c in id_cols if c in test_df.columns], inplace=True)

# 1-B  simple “count” features for |-separated lists
for col in MULTIVALUE_COLS:
    if col in df.columns:
        new = f"num_{col.replace(' ','_').lower()}"
        df[new]       = df[col].fillna("").str.split("|").apply(lambda x: len([t for t in x if t]))
        test_df[new]  = test_df.get(col, "").fillna("").str.split("|").apply(lambda x: len([t for t in x if t]))

# 1-C  procedure-code parsing + one-hot
if PROC_COL in df.columns:
    df["_proc_list"]   = df[PROC_COL].fillna("").apply(lambda s:[c.strip() for c in str(s).split(",") if c.strip()])
    test_df["_proc_list"] = test_df[PROC_COL].fillna("").apply(lambda s:[c.strip() for c in str(s).split(",") if c.strip()])

    df["num_procedure_codes"]   = df["_proc_list"].apply(len)
    test_df["num_procedure_codes"] = test_df["_proc_list"].apply(len)

    counter   = collections.Counter(code for codes in df["_proc_list"] for code in codes)
    top_codes = [c for c, _ in counter.most_common(TOP_PROC_K)]

    for code in top_codes:
        safe = code.replace(".", "_")
        df[f"has_proc_{safe}"]   = df["_proc_list"].apply(lambda lst: int(code in lst))
        test_df[f"has_proc_{safe}"] = test_df["_proc_list"].apply(lambda lst: int(code in lst))

    df.drop(columns=[PROC_COL, "_proc_list"], inplace=True)
    test_df.drop(columns=[PROC_COL, "_proc_list"], inplace=True)
# ╰─────────────────────────────────────────────────────────────╯

# ╭────────────────── 2. CATEGORICAL / NUMERIC SPLIT ───────────╮
cat_cols, num_cols = [], []
for col in df.columns:
    if col == TARGET: continue
    if df[col].dtype == "object":
        top_vals = df[col].value_counts().nlargest(KEEP_TOP_K).index
        df[col]      = np.where(df[col].isin(top_vals), df[col], "UNK")
        test_df[col] = np.where(test_df[col].isin(top_vals), test_df[col], "UNK")
        cat_cols.append(col)
    elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= SMALL_CARD:
        cat_cols.append(col)
    else:
        num_cols.append(col)

# fill missing
df[cat_cols]      = df[cat_cols].fillna("UNK")
test_df[cat_cols] = test_df[cat_cols].fillna("UNK")
df[num_cols]      = df[num_cols].fillna(df[num_cols].median())
test_df[num_cols] = test_df[num_cols].fillna(df[num_cols].median())

# drop near-constant numerics
if num_cols:
    vt = VarianceThreshold(1e-6).fit(df[num_cols])
    num_cols = [c for c, keep in zip(num_cols, vt.get_support()) if keep]
    df[num_cols]      = vt.transform(df[num_cols])
    test_df[num_cols] = vt.transform(test_df[num_cols])

# scale continuous numerics
to_scale   = [c for c in num_cols if df[c].nunique() > 2]
scaler_all = StandardScaler().fit(df[to_scale])
df[to_scale]      = scaler_all.transform(df[to_scale])
test_df[to_scale] = scaler_all.transform(test_df[to_scale])
# ╰─────────────────────────────────────────────────────────────╯

# ╭────────────────── 3. ENCODE CATEGORICALS ───────────────────╮
cat_maps, cat_dims = {}, {}
for col in cat_cols:
    codes, uniq  = pd.factorize(df[col], sort=True)
    df[col]      = codes + 1                    # 0 reserved for UNK
    cat_maps[col]= dict(enumerate(["UNK"]+list(uniq)))
    cat_dims[col]= int(df[col].max()) + 1

    test_df[col] = (
        test_df[col]
        .map({v: k for k, v in cat_maps[col].items()})
        .fillna(0)
        .astype(int)
    )
# ╰─────────────────────────────────────────────────────────────╯

# optional log-transform
if LOG_TARGET:
    df[TARGET]      = np.log1p(df[TARGET])
    test_df[TARGET] = np.log1p(test_df[TARGET])

y_for_strat = pd.qcut(df[TARGET], 4, labels=False, duplicates="drop").values

# ╭────────────────────── Dataset (uses iloc) ──────────────────╮
class TabDataset(Dataset):
    """idx – 1-D array of **row positions** (0…len-1)."""
    def __init__(self, idx, frame):
        self.idx = np.asarray(idx)
        self.Xn  = frame[num_cols].to_numpy(dtype=np.float32, copy=False)
        self.Xc  = frame[cat_cols].to_numpy(dtype=np.int64,   copy=False)
        self.y   = frame[TARGET].to_numpy(dtype=np.float32,   copy=False).reshape(-1,1)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(self.Xn[j]),
            torch.from_numpy(self.Xc[j]),
            torch.from_numpy(self.y[j])
        )
# ╰─────────────────────────────────────────────────────────────╯

def emb_size(card): return min(50, round(1.6 * card ** 0.25))

class SurgeryNet(nn.Module):
    def __init__(self, n_num, cat_dims, emb_factor, dropout):
        super().__init__()
        emb_dims = [max(1, int(emb_size(c) * emb_factor)) for c in cat_dims]
        self.embeds = nn.ModuleList([nn.Embedding(c, e) for c, e in zip(cat_dims, emb_dims)])
        in_dim = sum(emb_dims) + n_num
        layers = []
        for h in (256, 128, 64):
            layers += [nn.Linear(in_dim, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(True),
                       nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, xn, xc):
        embs = [e(xc[:, i]) for i, e in enumerate(self.embeds)]
        return self.mlp(torch.cat(embs + [xn], 1))

mse_loss = nn.MSELoss()

@torch.no_grad()
def metrics_on_loader(model, loader):
    model.eval(); p, t = [], []
    for xn, xc, yb in loader:
        p.extend(model(xn.to(device), xc.to(device)).cpu().numpy().ravel())
        t.extend(yb.numpy().ravel())

    if LOG_TARGET:
        p, t = np.expm1(p), np.expm1(t)

    # ★ save predictions on the model so we can retrieve them later
    model._last_preds = np.array(p)

    mae  = mean_absolute_error(t, p)
    mse  = mean_squared_error(t, p)
    rmse = math.sqrt(mse)
    r2   = r2_score(t, p)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2)


def mae_on_loader(model, loader):
    return metrics_on_loader(model, loader)["MAE"]

def run_epoch(model, loader, opt=None, scaler=None):
    train = opt is not None
    model.train() if train else model.eval()
    for xn, xc, yb in loader:
        xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pred = model(xn, xc)
            loss = mse_loss(pred, yb)
        if train:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

# ╭────────────────────── Optuna objective ─────────────────────╮
def objective(trial):
    params = dict(
        dropout     = trial.suggest_float("dropout", 0.1, 0.5),
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        wd          = trial.suggest_float("wd", 1e-7, 1e-4, log=True),
        batch       = trial.suggest_categorical("batch", [32, 64, 128]),
        emb_factor  = trial.suggest_float("emb_factor", 0.7, 2.0)
    )

    skf  = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    maes = []
    for tr_idx, va_idx in skf.split(df, y_for_strat):
        model = SurgeryNet(len(num_cols), [cat_dims[c] for c in cat_cols],
                           params["emb_factor"], params["dropout"]).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        tr_ld = DataLoader(TabDataset(tr_idx, df), batch_size=params["batch"],
                           shuffle=True, pin_memory=(device.type == "cuda"))
        va_ld = DataLoader(TabDataset(va_idx, df), batch_size=256,
                           pin_memory=(device.type == "cuda"))

        best, wait = 1e9, 0
        for _ in range(MAX_EPOCHS):
            run_epoch(model, tr_ld, opt, scaler)
            m = mae_on_loader(model, va_ld)
            if m < best - 0.05: best, wait = m, 0
            else:                wait += 1
            if wait >= PATIENCE: break
        maes.append(best)
        trial.report(best, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(maes)
# ╰─────────────────────────────────────────────────────────────╯

warnings.filterwarnings("ignore", category=UserWarning)
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best = study.best_params
print("Best hyper-parameters:", best)

# ╭────────────────── 5. Retrain on *all* train rows ───────────╮
full_ld = DataLoader(
    TabDataset(np.arange(len(df)), df),
    batch_size=best["batch"], shuffle=True, pin_memory=(device.type == "cuda")
)

model = SurgeryNet(len(num_cols), [cat_dims[c] for c in cat_cols],
                   best["emb_factor"], best["dropout"]).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["wd"])
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

best_state, best_mae, wait = None, 1e9, 0
for ep in range(1, MAX_EPOCHS + 1):
    run_epoch(model, full_ld, opt, scaler)
    sched.step()

    samp_idx = np.random.choice(len(df), int(0.15 * len(df)), replace=False)
    m = mae_on_loader(model,
                      DataLoader(TabDataset(samp_idx, df), batch_size=256,
                                 pin_memory=(device.type == "cuda")))
    if m < best_mae - 0.05:
        best_mae, wait, best_state = m, 0, model.state_dict()
    else:
        wait += 1
    if wait >= PATIENCE:
        break
    print(f"Epoch {ep:3d}  MAE={m:6.2f}   best={best_mae:6.2f}")

print(f"Final held-out 15 %-inside-train MAE: {best_mae:.2f} min")
model.load_state_dict(best_state)

# ╭────────────────────── 6. FINAL TEST ────────────────────────╮
test_loader = DataLoader(
    TabDataset(np.arange(len(test_df)), test_df),
    batch_size=256, pin_memory=(device.type == "cuda")
)
test_metrics = metrics_on_loader(model, test_loader)
print("\n🧪 15 % hold-out **test** results:")
print(f"MAE: {test_metrics['MAE']:.2f}   RMSE: {test_metrics['RMSE']:.2f}   R²: {test_metrics['R2']:.4f}")

# ╭────────────────────── 7. SAVE ARTEFACTS ────────────────────╮
torch.save(
    {
        "model_state": best_state,
        "scaler_num":  scaler_all,
        "cat_maps":    cat_maps,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "log_target":  LOG_TARGET,
        "params":      best
    },
    "best_surgery_model_v4.pt"
)
with open("best_model_info_v4.txt", "w") as f:
    json.dump({"MAE": float(test_metrics["MAE"]), "best_params": best}, f, indent=2)

print("\n✅ Model, scaler & metadata saved.")
