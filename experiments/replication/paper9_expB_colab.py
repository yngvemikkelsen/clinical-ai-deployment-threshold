# =====================================================================
# Paper 9 - Experiment B on Colab
# Fine-tune CodeBERT (6 languages) + Code Llama embeddings.
# Paste into ONE Colab cell. Runtime > Change runtime type > GPU first.
#
# Everything is written to Drive as it completes, so a disconnect costs
# only the language in flight. Re-running the cell skips finished work.
#
# Local Mac already produced and validated: codebert + codet5p embeddings,
# 12/12 primary cells within 0.001 of Diera Table 2. This fills in the two
# remaining models - fine-tuned CodeBERT (harmed) and Code Llama (benefits).
# =====================================================================

HF_TOKEN = ""          # <<< PASTE YOUR TOKEN HERE
LANGS    = ["ruby", "javascript", "go", "java", "python", "php"]
DO_FINETUNE = True
DO_CODELLAMA = True
FT_EPOCHS, FT_BS, FT_LR = 5, 32, 5e-5
LLAMA_BS = 8
MAX_LEN = 256
SEED = 42

# ---------------------------------------------------------------- setup
import os, sys, json, time, gc, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "datasets", "huggingface_hub", "pandas",
                "pyarrow", "info-nce-pytorch", "accelerate"], check=True)

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, HfApi, login
from tqdm.auto import tqdm

assert HF_TOKEN, "paste your HF token into HF_TOKEN above"
login(token=HF_TOKEN, add_to_git_credential=False)
os.environ["HF_TOKEN"] = HF_TOKEN

from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/paper9_expB"
MODELS, EMBS = f"{OUT}/models", f"{OUT}/embeddings"
for d in (OUT, MODELS, EMBS):
    os.makedirs(d, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "no GPU - set Runtime > Change runtime type > GPU"
print(torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

torch.manual_seed(SEED); np.random.seed(SEED)
CSN = "code-search-net/code_search_net"


def csn(lang, split):
    """Parquet conversion; the old loading script is gone."""
    files = sorted(f for f in HfApi().list_repo_files(CSN, repo_type="dataset")
                   if f.startswith(f"{lang}/{split}-"))
    if not files:
        raise FileNotFoundError(f"{lang}/{split}")
    return pd.concat([pd.read_parquet(
        hf_hub_download(repo_id=CSN, repo_type="dataset", filename=f))
        for f in files], ignore_index=True)


# ------------------------------------------------- fine-tune CodeBERT
class PairDS(Dataset):
    """Mirrors the original repo: add_special_tokens=False, pad to max_length.

    NOTE their fine-tuning tokenises WITHOUT special tokens while
    create_embeddings.py uses the default (with). That asymmetry is theirs;
    it is preserved deliberately so the replication stays faithful.
    """
    def __init__(self, df, tok, max_len=MAX_LEN):
        self.doc = [" ".join(x).strip() if not isinstance(x, str) else x
                    for x in df["func_documentation_tokens"]]
        self.code = [" ".join(x).strip() if not isinstance(x, str) else x
                     for x in df["func_code_tokens"]]
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, i):
        d = self.tok(" ".join(str(self.doc[i]).split()),
                     add_special_tokens=False, max_length=self.max_len,
                     truncation=True, padding="max_length",
                     return_token_type_ids=False)
        c = self.tok(str(self.code[i]),
                     add_special_tokens=False, max_length=self.max_len,
                     truncation=True, padding="max_length",
                     return_token_type_ids=False)
        return {"doc_ids": torch.tensor(d["input_ids"]),
                "doc_mask": torch.tensor(d["attention_mask"]),
                "code_ids": torch.tensor(c["input_ids"]),
                "code_mask": torch.tensor(c["attention_mask"])}


def mean_pool(h, m):
    m = m.unsqueeze(-1).expand(h.size()).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


def finetune(lang):
    dst = f"{MODELS}/codebert_{lang}.pth"
    if os.path.exists(dst):
        print(f"  {lang}: already done, skipping"); return
    from info_nce import InfoNCE
    df = csn(lang, "train")
    tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(DEV)
    dl = DataLoader(PairDS(df, tok), batch_size=FT_BS, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    loss_fn, scaler = InfoNCE(), torch.cuda.amp.GradScaler()
    print(f"  {lang}: {len(df):,} pairs, {len(dl):,} batches x {FT_EPOCHS} epochs")
    model.train(); t0 = time.time()
    for ep in range(FT_EPOCHS):
        tot = 0.0
        for b in tqdm(dl, desc=f"{lang} ep{ep+1}", leave=False):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                q = mean_pool(model(input_ids=b["doc_ids"].to(DEV),
                                    attention_mask=b["doc_mask"].to(DEV)
                                    ).last_hidden_state, b["doc_mask"].to(DEV))
                k = mean_pool(model(input_ids=b["code_ids"].to(DEV),
                                    attention_mask=b["code_mask"].to(DEV)
                                    ).last_hidden_state, b["code_mask"].to(DEV))
                loss = loss_fn(q, k)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"    ep{ep+1} loss {tot/len(dl):.4f} "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
    torch.save(model.state_dict(), dst)
    print(f"  {lang}: saved -> {dst}")
    del model, dl; gc.collect(); torch.cuda.empty_cache()


# -------------------------------------------------------- embeddings
def texts(df, kind):
    col = "func_documentation_tokens" if kind == "doc" else "func_code_tokens"
    return [x.strip() if isinstance(x, str) else " ".join(x).strip()
            for x in df[col]]


def embed(name, ckp, lang, bs, ft_state=None, fp16=False):
    """Batched, masked mean pooling - matches the validated local pipeline."""
    tag = "_finetuned" if ft_state else ""
    if all(os.path.exists(f"{EMBS}/{k}_embs_{name}_{lang}{tag}.npy")
           for k in ("code", "doc")):
        print(f"  {name}/{lang}: cached, skipping"); return
    tok = AutoTokenizer.from_pretrained(ckp, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(ckp, trust_remote_code=True)
    for a, v in (("is_decoder", False), ("is_encoder_decoder", False),
                 ("use_cache", False), ("tie_word_embeddings", False)):
        if not hasattr(cfg, a):
            setattr(cfg, a, v)
    kw = dict(config=cfg, trust_remote_code=True)
    if fp16:
        kw["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(ckp, **kw)
    if ft_state:
        model.load_state_dict(torch.load(ft_state, map_location="cpu"))
    model.to(DEV).eval()

    df = csn(lang, "test")
    for kind in ("code", "doc"):
        T, out = texts(df, kind), []
        for i in tqdm(range(0, len(T), bs), desc=f"{name}/{lang}/{kind}",
                      leave=False):
            enc = tok(T[i:i+bs], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEV)
            with torch.no_grad():
                o = model(**enc)
            if hasattr(o, "last_hidden_state"):
                p = mean_pool(o.last_hidden_state, enc["attention_mask"])
            else:
                p = o if isinstance(o, torch.Tensor) else o[0]
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                p = p.reshape(p.shape[0], -1)
            out.append(p.detach().cpu().float().numpy())
        arr = np.concatenate(out, 0)
        np.save(f"{EMBS}/{kind}_embs_{name}_{lang}{tag}.npy", arr)
        print(f"  {name}/{lang}/{kind}: {arr.shape}")
    del model; gc.collect(); torch.cuda.empty_cache()


# ------------------------------------------------------------- run
t_start = time.time()

if DO_FINETUNE:
    print("\n=== FINE-TUNING CodeBERT ===")
    for L in LANGS:
        finetune(L)
    print("\n=== FT-CodeBERT embeddings ===")
    for L in LANGS:
        embed("codebert", "microsoft/codebert-base", L, 32,
              ft_state=f"{MODELS}/codebert_{L}.pth")

if DO_CODELLAMA:
    print("\n=== Code Llama embeddings ===")
    for L in LANGS + ["r"]:
        if L == "r":
            print("  r: run locally (bundled statcodesearch jsonl)"); continue
        embed("codellama", "codellama/CodeLlama-7b-hf", L, LLAMA_BS, fp16=True)

print(f"\nDONE in {(time.time()-t_start)/60:.0f} min")
print(f"artefacts in {OUT}")
print("\nNext, locally:")
print("  copy models/*.pth and embeddings/*.npy into code_isotropy/")
print("  python paper9_expB_diera.py embed --lang r --model codellama")
print("  python paper9_expB_diera.py evaluate --all")
print("  python paper9_expB_diera.py threshold")
