#!/usr/bin/env python3
"""Stage 2: Null-controlled Geneformer-V2-104M
single-gene, combinatorial, and overexpression virtual perturbations in patient-balanced
ER+ malignant breast epithelial cells (Wu et al. GSE176078).

Geneformer-V2-104M configuration (Theodoris et al. 2023):
  - Median-rescaled rank-value tokenization: (UMI / total_cell_UMI * 1e4) / gene_median_dictionary_gc104M
  - Special tokens: [<cls>] + top expressed genes (up to 4,094) + [<eos>], MAX_LEN = 4096
  - Penultimate hidden layer (hidden_states[-2], layer 11 of 12; emb_layer = -1)
  - Deletion (token removal) & overexpression (move/insert target token at rank 1 after <cls>)
  - Patient-balanced Core ER+ Pool: 50 HLA-A+B2M+ ER+ cells per patient x 9 ER+ patients = 450 cells
  - 115 rank- and cardinality-matched empirical nulls:
      40 single_del (P_min = 0.0244), 25 combo2_del (P_min = 0.0385),
      25 combo3_del (P_min = 0.0385), 25 oe (P_min = 0.0385)

Outputs under results/stage2_geneformer/:
  - er_core_pool_cells.csv
  - gf_<pert_name>_gene_shifts.csv
  - gf_null_summary.csv
  - gf_perturbation_summary.json
"""
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.io import mmread
from transformers import BertForMaskedLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WU = os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq")
DICTS = os.path.expanduser("~/.cache/geneformer/gf_dicts/geneformer")
MODEL_DIR = os.path.expanduser("~/.cache/geneformer/Geneformer-V2-104M")
OUT = os.path.join(ROOT, "results/stage2_geneformer")
os.makedirs(OUT, exist_ok=True)

SEED = 20261119
PAD_ID = 0
CLS_ID = 2
EOS_ID = 3
MAX_LEN = 4096          # Geneformer-V2-104M max_position_embeddings
EMB_LAYER_IDX = -2      # Penultimate layer (layer 11 of 12, emb_layer=-1)
BATCH_SIZE = 8

APP4 = ["HLA-A", "B2M", "TAP1", "PSMB8"]
APP13 = [
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DRB5", "HLA-DPA1",
    "HLA-DPB1", "CD74", "PSMB9", "PSMB8", "TAP1", "TAP2", "B2M"
]
HK_GENES = ["ACTB", "GAPDH", "RPLP0", "HPRT1"]
EXCLUDE_NULL = set(APP13 + HK_GENES + [
    "HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC4", "HDAC5", "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11",
    "ESR1", "FOXA1", "GATA3", "JAK1", "JAK2", "STAT1", "IRF1", "NLRC5", "CIITA"
])


def encode_cells_v2(C, lib_all, sym_k, tok_k, med_k, max_len=MAX_LEN):
    """Geneformer-V2 median-rescaled rank-value tokenization with <cls> and <eos>."""
    max_genes = max_len - 2
    ids_list, syms_list = [], []
    for i in range(C.shape[0]):
        row = C[i]
        nz = np.where(row > 0)[0]
        if len(nz) == 0:
            ids_list.append([CLS_ID, EOS_ID])
            syms_list.append([])
            continue
        norm_vals = (row[nz] / max(float(lib_all[i]), 1.0) * 1e4) / med_k[nz]
        order = nz[np.argsort(-norm_vals, kind="stable")][:max_genes]
        ids_list.append([CLS_ID] + [int(tok_k[k]) for k in order] + [EOS_ID])
        syms_list.append([sym_k[k] for k in order])
    return ids_list, syms_list


@torch.no_grad()
def forward_batched_v2(bert_model, token_lists, dev, batch_size=BATCH_SIZE):
    """Length-sorted batched forward pass returning penultimate layer hidden states (CPU float32)
    and mean-pooled cell embeddings (excluding <cls> and <eos>)."""
    n = len(token_lists)
    order = sorted(range(n), key=lambda i: len(token_lists[i]))
    all_h = [None] * n
    cell_embs = [None] * n
    for s in range(0, n, batch_size):
        idxs = order[s: s + batch_size]
        mlen = max(len(token_lists[i]) for i in idxs)
        ids = torch.full((len(idxs), mlen), PAD_ID, dtype=torch.long, device=dev)
        mask = torch.zeros((len(idxs), mlen), dtype=torch.long, device=dev)
        for bi, ci in enumerate(idxs):
            seq = token_lists[ci]
            L = len(seq)
            ids[bi, :L] = torch.tensor(seq, dtype=torch.long, device=dev)
            mask[bi, :L] = 1
        out = bert_model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[EMB_LAYER_IDX]
        for bi, ci in enumerate(idxs):
            L = len(token_lists[ci])
            h_seq = out[bi, :L].detach().cpu()
            all_h[ci] = h_seq
            # Mean over expressed gene tokens (excluding <cls> at 0 and <eos> at L-1)
            cell_embs[ci] = h_seq[1:L - 1].mean(dim=0) if L > 2 else h_seq[0]
    return all_h, torch.stack(cell_embs, dim=0)


def apply_perturbation_v2(ids_wt, syms_wt, mode, targets, sym2tok, max_len=MAX_LEN):
    """Geneformer-V2 delete and overexpress (move/insert at rank 1 after <cls>)."""
    tset = set(targets)
    if mode == "delete":
        new_ids = [CLS_ID]
        new_syms = []
        for s, tid in zip(syms_wt, ids_wt[1:-1]):
            if s not in tset:
                new_ids.append(tid)
                new_syms.append(s)
        new_ids.append(EOS_ID)
        return new_ids, new_syms
    elif mode == "oe":
        # perturber_utils.overexpress_tokens: remove existing occurrence(s),
        # insert target token(s) at index 1 immediately after <cls>, preserve <eos> at end.
        rem_ids, rem_syms = [], []
        for s, tid in zip(syms_wt, ids_wt[1:-1]):
            if s not in tset:
                rem_ids.append(tid)
                rem_syms.append(s)
        ins_ids = [int(sym2tok[t]) for t in targets]
        ins_syms = list(targets)
        max_genes = max_len - 2
        comb_ids = (ins_ids + rem_ids)[:max_genes]
        comb_syms = (ins_syms + rem_syms)[:max_genes]
        return [CLS_ID] + comb_ids + [EOS_ID], comb_syms
    raise ValueError(mode)


def patient_balanced_subsample(indices, pats_arr, max_n, rng):
    """Subsample up to max_n indices uniformly stratified across patients."""
    if len(indices) <= max_n:
        return list(indices)
    by_pat = {}
    for idx in indices:
        by_pat.setdefault(pats_arr[idx], []).append(idx)
    for p in by_pat:
        rng.shuffle(by_pat[p])
    chosen = []
    pat_keys = sorted(by_pat.keys())
    pos = {p: 0 for p in pat_keys}
    while len(chosen) < max_n:
        progressed = False
        for p in pat_keys:
            if pos[p] < len(by_pat[p]) and len(chosen) < max_n:
                chosen.append(by_pat[p][pos[p]])
                pos[p] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def score_perturbation_v2(
    bert_model, dev, h_wt_list, emb_wt, ids_wt_list, syms_wt_list,
    mode, targets, sym2tok, centroid_norm, centroid_tnbc,
    cell_subset=None, save_gene_df=True
):
    """Run perturbed forward pass and score gene displacements + cell-state directional shifts."""
    tset = set(targets)
    cand_indices = list(range(len(syms_wt_list))) if cell_subset is None else list(cell_subset)
    pt_ids_list, pt_syms_list, valid_idx = [], [], []
    for ci in cand_indices:
        syms = syms_wt_list[ci]
        if mode == "delete":
            if not tset.issubset(set(syms)):
                continue
        p_ids, p_syms = apply_perturbation_v2(ids_wt_list[ci], syms, mode, targets, sym2tok)
        pt_ids_list.append(p_ids)
        pt_syms_list.append(p_syms)
        valid_idx.append(ci)

    if not valid_idx:
        return None, None

    h_pt_list, emb_pt = forward_batched_v2(bert_model, pt_ids_list, dev)

    acc = {}
    t_cos_norm_wt, t_cos_norm_pt = [], []
    t_cos_tnbc_wt, t_cos_tnbc_pt = [], []

    for out_i, ci in enumerate(valid_idx):
        h_w = h_wt_list[ci]
        h_p = h_pt_list[out_i]
        ids_w = ids_wt_list[ci]
        syms_w = syms_wt_list[ci]
        ids_p = pt_ids_list[out_i]

        # Token -> position in perturbed sequence (excluding <cls> at 0 and <eos> at len-1)
        pt_pos = {}
        for pos in range(1, len(ids_p) - 1):
            tid = ids_p[pos]
            if tid not in pt_pos:
                pt_pos[tid] = pos

        w_idx, p_idx, g_syms = [], [], []
        for pos_w in range(1, len(ids_w) - 1):
            s = syms_w[pos_w - 1]
            if s in tset:
                continue
            tid = ids_w[pos_w]
            if tid in pt_pos:
                w_idx.append(pos_w)
                p_idx.append(pt_pos[tid])
                g_syms.append(s)

        if w_idx:
            cos_vals = torch.nn.functional.cosine_similarity(
                h_w[w_idx], h_p[p_idx], dim=1
            ).numpy()
            shifts = 1.0 - cos_vals
            for s, d in zip(g_syms, shifts):
                cur = acc.get(s)
                if cur is None:
                    acc[s] = [float(d), 1]
                else:
                    cur[0] += float(d)
                    cur[1] += 1

        ew = emb_wt[ci]
        ep = emb_pt[out_i]
        t_cos_norm_wt.append(torch.nn.functional.cosine_similarity(ew, centroid_norm, dim=0).item())
        t_cos_norm_pt.append(torch.nn.functional.cosine_similarity(ep, centroid_norm, dim=0).item())
        t_cos_tnbc_wt.append(torch.nn.functional.cosine_similarity(ew, centroid_tnbc, dim=0).item())
        t_cos_tnbc_pt.append(torch.nn.functional.cosine_similarity(ep, centroid_tnbc, dim=0).item())

    gene_df = pd.DataFrame(
        [(g, v[0] / v[1], v[1]) for g, v in acc.items()],
        columns=["gene", "mean_shift", "n_cells"]
    ).sort_values("mean_shift", ascending=False).reset_index(drop=True)
    gene_df["percentile"] = gene_df["mean_shift"].rank(pct=True)

    g_map = gene_df.set_index("gene")
    app4_vals = [float(g_map.loc[g, "mean_shift"]) for g in APP4 if g in g_map.index]
    app4_pcts = [float(g_map.loc[g, "percentile"]) for g in APP4 if g in g_map.index]
    app13_vals = [float(g_map.loc[g, "mean_shift"]) for g in APP13 if g in g_map.index]
    app13_pcts = [float(g_map.loc[g, "percentile"]) for g in APP13 if g in g_map.index]
    hk_vals = [float(g_map.loc[g, "mean_shift"]) for g in HK_GENES if g in g_map.index]

    delta_norm = np.array(t_cos_norm_pt) - np.array(t_cos_norm_wt)
    delta_tnbc = np.array(t_cos_tnbc_pt) - np.array(t_cos_tnbc_wt)

    summary = {
        "n_cells": len(valid_idx),
        "n_genes_scored": len(gene_df),
        "global_mean_shift": float(gene_df["mean_shift"].mean()),
        "app4_mean_shift": float(np.mean(app4_vals)) if app4_vals else np.nan,
        "app4_mean_pct": float(np.mean(app4_pcts)) if app4_pcts else np.nan,
        "app13_mean_shift": float(np.mean(app13_vals)) if app13_vals else np.nan,
        "app13_mean_pct": float(np.mean(app13_pcts)) if app13_pcts else np.nan,
        "hk_mean_shift": float(np.mean(hk_vals)) if hk_vals else np.nan,
        "per_gene_app4": {g: float(g_map.loc[g, "mean_shift"]) for g in APP4 if g in g_map.index},
        "per_gene_app4_pct": {g: float(g_map.loc[g, "percentile"]) for g in APP4 if g in g_map.index},
        "delta_cos_normal": float(delta_norm.mean()),
        "delta_cos_tnbc": float(delta_tnbc.mean()),
    }
    return (gene_df if save_gene_df else None), summary


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {dev} | Geneformer-V2-104M (median-rescaled, MAX_LEN={MAX_LEN}, layer={EMB_LAYER_IDX})")

    meta = pd.read_csv(os.path.join(WU, "metadata.csv"), index_col=0)
    genes = [l.strip() for l in open(os.path.join(WU, "count_matrix_genes.tsv"))]
    barcodes = [l.strip() for l in open(os.path.join(WU, "count_matrix_barcodes.tsv"))]
    X = mmread(os.path.join(WU, "count_matrix_sparse.mtx")).T.tocsr()
    lib_total = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    nfeat_total = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int32)
    g2i = {g: i for i, g in enumerate(genes)}

    tok_dict = pickle.load(open(os.path.join(DICTS, "token_dictionary_gc104M.pkl"), "rb"))
    name_dict = pickle.load(open(os.path.join(DICTS, "gene_name_id_dict_gc104M.pkl"), "rb"))
    med_dict = pickle.load(open(os.path.join(DICTS, "gene_median_dictionary_gc104M.pkl"), "rb"))
    special = {0, 1, 2, 3}

    keep_quads = []
    for j, g in enumerate(genes):
        ens = name_dict.get(g)
        if ens and ens in tok_dict and ens in med_dict:
            t = tok_dict[ens]
            if t not in special:
                keep_quads.append((j, g, int(t), float(med_dict[ens])))
    col_j = [x[0] for x in keep_quads]
    sym_k = [x[1] for x in keep_quads]
    tok_k = np.array([x[2] for x in keep_quads], dtype=np.int64)
    med_k = np.array([x[3] for x in keep_quads], dtype=np.float32)
    sym2tok = {g: int(t) for _, g, t, _ in keep_quads}
    print(f"Mapped {len(sym_k)}/{len(genes)} genes to V2 token + median dictionaries.")

    bert_model = BertForMaskedLM.from_pretrained(MODEL_DIR).bert.to(dev).eval()

    col_hla = X[:, g2i["HLA-A"]].toarray().ravel() > 0
    col_b2m = X[:, g2i["B2M"]].toarray().ravel() > 0
    col_psmb8 = X[:, g2i["PSMB8"]].toarray().ravel() > 0
    ct_series = meta["celltype_major"].reindex(barcodes).values
    st_series = meta["subtype"].reindex(barcodes).values
    pat_series = meta["orig.ident"].reindex(barcodes).values

    # 1. Patient-balanced Normal Epithelial & High-APP TNBC reference centroids
    norm_idx = np.where((ct_series == "Normal Epithelial") & col_hla & col_b2m)[0]
    norm_sel = patient_balanced_subsample(norm_idx, pat_series, 200, rng)
    C_norm = X[norm_sel, :][:, col_j].toarray().astype(np.float32)
    ids_n, _ = encode_cells_v2(C_norm, lib_total[norm_sel], sym_k, tok_k, med_k)
    _, emb_norm = forward_batched_v2(bert_model, ids_n, dev)
    centroid_norm = emb_norm.mean(dim=0)

    tnbc_idx = np.where((ct_series == "Cancer Epithelial") & (st_series == "TNBC") & col_hla & col_b2m & col_psmb8)[0]
    tnbc_sel = patient_balanced_subsample(tnbc_idx, pat_series, 200, rng)
    C_tnbc = X[tnbc_sel, :][:, col_j].toarray().astype(np.float32)
    ids_t, _ = encode_cells_v2(C_tnbc, lib_total[tnbc_sel], sym_k, tok_k, med_k)
    _, emb_tnbc = forward_batched_v2(bert_model, ids_t, dev)
    centroid_tnbc = emb_tnbc.mean(dim=0)
    print(f"Reference centroids built: Normal n={len(norm_sel)}, High-APP TNBC n={len(tnbc_sel)} ({time.time()-t0:.1f}s)")

    # 2. Patient-balanced ER+ Core Pool: exactly 50 HLA-A+B2M+ ER+ cells per patient x 9 patients = 450 cells
    er_all_idx = np.where((ct_series == "Cancer Epithelial") & (st_series == "ER+"))[0]
    er_elig = er_all_idx[col_hla[er_all_idx] & col_b2m[er_all_idx]]
    core_sel = []
    for p in sorted(pd.unique(pat_series[er_elig])):
        p_idx = er_elig[pat_series[er_elig] == p]
        k = min(50, len(p_idx))
        core_sel.extend(rng.choice(p_idx, size=k, replace=False).tolist())
    core_sel = sorted(core_sel)
    core_pats = np.array([pat_series[i] for i in core_sel])

    C_core = X[core_sel, :][:, col_j].toarray().astype(np.float32)
    ids_core, syms_core = encode_cells_v2(C_core, lib_total[core_sel], sym_k, tok_k, med_k)
    h_wt_core, emb_wt_core = forward_batched_v2(bert_model, ids_core, dev)

    core_df = pd.DataFrame({
        "barcode": [barcodes[i] for i in core_sel],
        "mtx_idx": core_sel,
        "patient": core_pats,
        "n_umi": lib_total[core_sel].astype(int),
        "n_genes": nfeat_total[core_sel].astype(int),
        "seq_len": [len(s) for s in ids_core],
    })
    core_df.to_csv(os.path.join(OUT, "er_core_pool_cells.csv"), index=False)
    print(
        f"Patient-balanced ER+ Core Pool: n={len(core_sel)} across {core_df['patient'].nunique()} patients "
        f"(max pat share={core_df['patient'].value_counts().max()/len(core_sel)*100:.1f}%, "
        f"med UMI={int(core_df['n_umi'].median())} vs all ER+ {int(np.median(lib_total[er_all_idx]))}, "
        f"med seq_len={int(core_df['seq_len'].median())}) [{time.time()-t0:.1f}s]"
    )

    # Compute per-gene median V2 token rank and presence rate across the 450 Core Pool cells
    rank_lists = {}
    for syms in syms_core:
        for r_pos, g in enumerate(syms, 1):
            rank_lists.setdefault(g, []).append(r_pos)
    gene_v2_med_rank = {g: float(np.median(v)) for g, v in rank_lists.items()}
    gene_v2_det = {g: len(v) / len(syms_core) for g, v in rank_lists.items()}

    # Print key gene ranks under V2 median-rescaled tokenization
    print("V2 median ranks in 450-cell balanced ER+ Core Pool:")
    for g in ["B2M", "HLA-A", "TAP1", "PSMB8", "ESR1", "HDAC1", "HDAC2", "HDAC3", "JAK1", "STAT1", "IRF1", "NLRC5"]:
        print(f"  {g:6s}: med_rank={gene_v2_med_rank.get(g, np.nan):7.1f} | det_in_pool={gene_v2_det.get(g, 0)*100:5.1f}%")

    # Patient-balanced 180-cell subset (20/patient x 9) for rank-1 OE perturbations
    oe_sub_idx = patient_balanced_subsample(list(range(len(core_sel))), core_pats, 180, rng)

    core_perts = [
        ("del_HDAC1", "delete", ["HDAC1"], None),
        ("del_HDAC2", "delete", ["HDAC2"], None),
        ("del_HDAC3", "delete", ["HDAC3"], None),
        ("del_HDAC1_HDAC2", "delete", ["HDAC1", "HDAC2"], None),
        ("del_HDAC1_HDAC2_HDAC3", "delete", ["HDAC1", "HDAC2", "HDAC3"], None),
        ("del_JAK1", "delete", ["JAK1"], None),
        ("del_STAT1", "delete", ["STAT1"], None),
        ("del_ESR1", "delete", ["ESR1"], None),
        ("oe_STAT1", "oe", ["STAT1"], oe_sub_idx),
        ("oe_IRF1", "oe", ["IRF1"], oe_sub_idx),
        ("oe_NLRC5", "oe", ["NLRC5"], oe_sub_idx),
    ]

    results = {}
    for pname, mode, targets, csub in core_perts:
        gdf, summ = score_perturbation_v2(
            bert_model, dev, h_wt_core, emb_wt_core, ids_core, syms_core,
            mode, targets, sym2tok, centroid_norm, centroid_tnbc,
            cell_subset=csub, save_gene_df=True
        )
        gdf.to_csv(os.path.join(OUT, f"gf_{pname}_gene_shifts.csv"), index=False)
        summ["pool"] = "balanced_core_450" if csub is None else f"balanced_core_{len(csub)}"
        summ["mode"] = mode
        summ["targets"] = targets
        summ["target_median_v2_ranks"] = {t: gene_v2_med_rank.get(t, None) for t in targets}
        results[pname] = summ
        print(
            f"  {pname:25s}: n={summ['n_cells']:3d} | APP4_shift={summ['app4_mean_shift']:.5f} "
            f"(pct={summ['app4_mean_pct']*100:4.1f}%) | HK={summ['hk_mean_shift']:.5f} "
            f"| dNorm={summ['delta_cos_normal']:+.5f} [{time.time()-t0:.0f}s]"
        )

    # 3. Patient-capped Secondary Pools for low-detection JAK2 and JAK1+JAK2 (<=40 cells/patient)
    Xer = X[er_all_idx, :]
    sec_specs = [
        ("del_JAK2", "delete", ["JAK2"], ["JAK2", "HLA-A", "B2M"], 200),
        ("del_JAK1_JAK2", "delete", ["JAK1", "JAK2"], ["JAK1", "JAK2", "HLA-A", "B2M"], 180),
    ]
    for pname, mode, targets, req, nmax in sec_specs:
        m_s = np.ones(len(er_all_idx), dtype=bool)
        for g in req:
            m_s &= (Xer[:, g2i[g]].toarray().ravel() > 0)
        elig = er_all_idx[m_s]
        sel = sorted(patient_balanced_subsample(elig, pat_series, nmax, rng))
        C_s = X[sel, :][:, col_j].toarray().astype(np.float32)
        ids_s, syms_s = encode_cells_v2(C_s, lib_total[sel], sym_k, tok_k, med_k)
        h_s, emb_s = forward_batched_v2(bert_model, ids_s, dev)
        gdf, summ = score_perturbation_v2(
            bert_model, dev, h_s, emb_s, ids_s, syms_s,
            mode, targets, sym2tok, centroid_norm, centroid_tnbc,
            save_gene_df=True
        )
        gdf.to_csv(os.path.join(OUT, f"gf_{pname}_gene_shifts.csv"), index=False)
        p_counts = pd.Series([pat_series[i] for i in sel]).value_counts()
        summ["pool"] = f"patient_balanced_sec_{len(sel)}"
        summ["n_patients"] = int(len(p_counts))
        summ["max_patient_share"] = float(p_counts.max() / len(sel))
        summ["mode"] = mode
        summ["targets"] = targets
        # compute target rank in this secondary pool
        sec_ranks = {t: [] for t in targets}
        for syms in syms_s:
            for r_pos, g in enumerate(syms, 1):
                if g in sec_ranks:
                    sec_ranks[g].append(r_pos)
        summ["target_median_v2_ranks"] = {t: float(np.median(v)) if v else None for t, v in sec_ranks.items()}
        results[pname] = summ
        print(
            f"  {pname:25s}: n={summ['n_cells']:3d} ({len(p_counts)} pats, max {p_counts.max()/len(sel)*100:.0f}%) "
            f"| APP4_shift={summ['app4_mean_shift']:.5f} (pct={summ['app4_mean_pct']*100:4.1f}%) "
            f"| dNorm={summ['delta_cos_normal']:+.5f} [{time.time()-t0:.0f}s]"
        )

    # 4. Truly Rank- and Cardinality-Matched Empirical Null Controls
    # Match the V2 median token rank & detection band of HDAC1/2/3/JAK1/STAT1 (excluding MT-/RPL/RPS)
    ref_targets = ["HDAC1", "HDAC2", "HDAC3", "JAK1", "STAT1"]
    ref_ranks = [gene_v2_med_rank[g] for g in ref_targets if g in gene_v2_med_rank]
    rank_lo = max(400.0, float(min(ref_ranks)) - 350.0)
    rank_hi = float(max(ref_ranks)) + 350.0
    print(f"\nConstructing rank-matched empirical null pool (target V2 ranks {min(ref_ranks):.0f}–{max(ref_ranks):.0f}; matched band [{rank_lo:.0f}, {rank_hi:.0f}])...")

    def is_clean_bg(g):
        if g in EXCLUDE_NULL:
            return False
        if g.startswith(("MT-", "RPL", "RPS", "MRPL", "MRPS")):
            return False
        return True

    # Single-del matched pool: rank inside [rank_lo, rank_hi], det in [0.22, 0.72]
    bg_single = sorted([
        g for g, r in gene_v2_med_rank.items()
        if is_clean_bg(g) and rank_lo <= r <= rank_hi and 0.22 <= gene_v2_det[g] <= 0.72
    ])
    # Combo2/Combo3 matched pool: same rank band, det in [0.35, 0.78] so 2- and 3-gene co-occurrence is well-populated
    bg_combo = sorted([
        g for g, r in gene_v2_med_rank.items()
        if is_clean_bg(g) and rank_lo <= r <= rank_hi and 0.35 <= gene_v2_det[g] <= 0.78
    ])
    print(f"Eligible rank-matched background genes: single/oe={len(bg_single)}, combo={len(bg_combo)}")

    # Pre-index which core cells express each bg gene
    cell_sets = {g: set() for g in set(bg_single) | set(bg_combo)}
    for ci, syms in enumerate(syms_core):
        s_set = set(syms)
        for g in s_set.intersection(cell_sets.keys()):
            cell_sets[g].add(ci)

    null_rows = []
    # 40 single-gene rank-matched nulls (up to 60 patient-balanced cells each) -> P_min = 1/41 = 0.0244
    null_single_genes = list(rng.choice(bg_single, size=40, replace=False))
    for k_i, ng in enumerate(null_single_genes, 1):
        elig_c = sorted(cell_sets[ng])
        csub = patient_balanced_subsample(elig_c, core_pats, 60, rng)
        _, summ = score_perturbation_v2(
            bert_model, dev, h_wt_core, emb_wt_core, ids_core, syms_core,
            "delete", [ng], sym2tok, centroid_norm, centroid_tnbc,
            cell_subset=csub, save_gene_df=False
        )
        null_rows.append({
            "null_type": "single_del", "targets": ng,
            "mean_v2_rank": gene_v2_med_rank[ng],
            **{k: summ[k] for k in ["n_cells", "app4_mean_shift", "app4_mean_pct", "app13_mean_shift", "app13_mean_pct", "hk_mean_shift", "delta_cos_normal", "delta_cos_tnbc", "global_mean_shift"]}
        })
        if k_i % 10 == 0:
            print(f"  single_del nulls: {k_i}/40 done [{time.time()-t0:.0f}s]")

    # 25 combo2-gene rank-matched nulls (up to 50 patient-balanced cells each) -> P_min = 1/26 = 0.0385
    done_c2 = 0
    tried = 0
    while done_c2 < 25 and tried < 5000:
        tried += 1
        pair = list(rng.choice(bg_combo, size=2, replace=False))
        elig_c = sorted(cell_sets[pair[0]] & cell_sets[pair[1]])
        if len(elig_c) < 25:
            continue
        csub = patient_balanced_subsample(elig_c, core_pats, 50, rng)
        _, summ = score_perturbation_v2(
            bert_model, dev, h_wt_core, emb_wt_core, ids_core, syms_core,
            "delete", pair, sym2tok, centroid_norm, centroid_tnbc,
            cell_subset=csub, save_gene_df=False
        )
        null_rows.append({
            "null_type": "combo2_del", "targets": "+".join(pair),
            "mean_v2_rank": float(np.mean([gene_v2_med_rank[g] for g in pair])),
            **{k: summ[k] for k in ["n_cells", "app4_mean_shift", "app4_mean_pct", "app13_mean_shift", "app13_mean_pct", "hk_mean_shift", "delta_cos_normal", "delta_cos_tnbc", "global_mean_shift"]}
        })
        done_c2 += 1
        if done_c2 % 5 == 0:
            print(f"  combo2_del nulls: {done_c2}/25 done [{time.time()-t0:.0f}s]")

    # 25 combo3-gene rank-matched nulls (up to 45 patient-balanced cells each) -> P_min = 1/26 = 0.0385
    done_c3 = 0
    tried = 0
    while done_c3 < 25 and tried < 10000:
        tried += 1
        trip = list(rng.choice(bg_combo, size=3, replace=False))
        elig_c = sorted(cell_sets[trip[0]] & cell_sets[trip[1]] & cell_sets[trip[2]])
        if len(elig_c) < 20:
            continue
        csub = patient_balanced_subsample(elig_c, core_pats, 45, rng)
        _, summ = score_perturbation_v2(
            bert_model, dev, h_wt_core, emb_wt_core, ids_core, syms_core,
            "delete", trip, sym2tok, centroid_norm, centroid_tnbc,
            cell_subset=csub, save_gene_df=False
        )
        null_rows.append({
            "null_type": "combo3_del", "targets": "+".join(trip),
            "mean_v2_rank": float(np.mean([gene_v2_med_rank[g] for g in trip])),
            **{k: summ[k] for k in ["n_cells", "app4_mean_shift", "app4_mean_pct", "app13_mean_shift", "app13_mean_pct", "hk_mean_shift", "delta_cos_normal", "delta_cos_tnbc", "global_mean_shift"]}
        })
        done_c3 += 1
        if done_c3 % 5 == 0:
            print(f"  combo3_del nulls: {done_c3}/25 done [{time.time()-t0:.0f}s]")

    # 25 rank-1 OE nulls on 60 patient-balanced Core Pool cells -> P_min = 1/26 = 0.0385
    oe_null_sub = patient_balanced_subsample(oe_sub_idx, core_pats, 60, rng)
    oe_pool = [g for g in bg_single if g not in set(null_single_genes)]
    null_oe_genes = list(rng.choice(oe_pool, size=25, replace=False))
    for k_i, ng in enumerate(null_oe_genes, 1):
        _, summ = score_perturbation_v2(
            bert_model, dev, h_wt_core, emb_wt_core, ids_core, syms_core,
            "oe", [ng], sym2tok, centroid_norm, centroid_tnbc,
            cell_subset=oe_null_sub, save_gene_df=False
        )
        null_rows.append({
            "null_type": "oe", "targets": ng,
            "mean_v2_rank": gene_v2_med_rank[ng],
            **{k: summ[k] for k in ["n_cells", "app4_mean_shift", "app4_mean_pct", "app13_mean_shift", "app13_mean_pct", "hk_mean_shift", "delta_cos_normal", "delta_cos_tnbc", "global_mean_shift"]}
        })
        if k_i % 5 == 0:
            print(f"  oe nulls: {k_i}/25 done [{time.time()-t0:.0f}s]")

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(os.path.join(OUT, "gf_null_summary.csv"), index=False)

    # Exact cardinality- and rank-matched empirical null P-values
    for pname, summ in results.items():
        if summ["mode"] == "oe":
            nt = "oe"
        elif len(summ["targets"]) == 3:
            nt = "combo3_del"
        elif len(summ["targets"]) == 2:
            nt = "combo2_del"
        else:
            nt = "single_del"
        nsub = null_df[null_df["null_type"] == nt]
        n_null = len(nsub)
        summ["matched_null_type"] = nt
        summ["n_nulls"] = int(n_null)
        summ["p_min_floor"] = float(1.0 / (1 + n_null))
        summ["null_median_v2_rank"] = float(nsub["mean_v2_rank"].median())
        summ["null_p_app4_shift"] = float((1 + (nsub["app4_mean_shift"] >= summ["app4_mean_shift"]).sum()) / (1 + n_null))
        summ["null_p_app4_pct"] = float((1 + (nsub["app4_mean_pct"] >= summ["app4_mean_pct"]).sum()) / (1 + n_null))
        summ["null_p_delta_normal"] = float((1 + (nsub["delta_cos_normal"] >= summ["delta_cos_normal"]).sum()) / (1 + n_null))
        summ["null_app4_shift_mean"] = float(nsub["app4_mean_shift"].mean())
        summ["null_app4_pct_mean"] = float(nsub["app4_mean_pct"].mean())
        summ["null_dnorm_mean"] = float(nsub["delta_cos_normal"].mean())

    # Save metadata block for reproducibility
    ns_s = null_df[null_df["null_type"] == "single_del"]
    r_rank_sh, p_rank_sh = stats.spearmanr(ns_s["mean_v2_rank"], ns_s["app4_mean_shift"])
    meta_out = {
        "spec": {
            "model": "Geneformer-V2-104M",
            "tokenization": "median_rescaled_CP10K",
            "special_tokens": ["<cls>", "<eos>"],
            "max_len": MAX_LEN,
            "emb_layer": "penultimate (hidden_states[-2], layer 11/12; emb_layer=-1)",
            "oe_operator": "rank1_insertion",
            "core_pool_n": len(core_sel),
            "core_pool_n_patients": int(core_df["patient"].nunique()),
            "core_pool_max_patient_share": float(core_df["patient"].value_counts().max() / len(core_sel)),
            "core_pool_median_umi": int(core_df["n_umi"].median()),
            "all_er_median_umi": int(np.median(lib_total[er_all_idx])),
            "null_counts": null_df["null_type"].value_counts().to_dict(),
            "null_rank_band": [rank_lo, rank_hi],
            "single_null_rank_vs_shift_rho": float(r_rank_sh),
            "single_null_rank_vs_shift_p": float(p_rank_sh),
            "key_gene_v2_median_ranks": {
                g: gene_v2_med_rank.get(g, None)
                for g in ["B2M", "HLA-A", "TAP1", "PSMB8", "ESR1", "HDAC1", "HDAC2", "HDAC3", "JAK1", "STAT1"]
            },
        },
        "perturbations": results,
    }
    # Preserve flat top-level perturbation keys for downstream compatibility + "_meta" key
    flat_out = dict(results)
    flat_out["_meta"] = meta_out["spec"]
    json.dump(flat_out, open(os.path.join(OUT, "gf_perturbation_summary.json"), "w"), indent=2)
    print(f"\n=== Stage 2 (Geneformer-V2-104M) Complete in {time.time()-t0:.1f}s ===")
    print("Null summary means:")
    print(null_df.groupby("null_type")[["mean_v2_rank", "n_cells", "app4_mean_shift", "app4_mean_pct"]].mean().round(5))



def _require_path(path: str, dataset_hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(
            f"[DATA REQUIRED] Expected raw input not found: {path}\n"
            f"  Dataset: {dataset_hint}\n"
            f"  See README.md ('Pipeline Reproduction') for public accession download paths."
        )

if __name__ == "__main__":
    _require_path(os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"), "Wu et al. GSE176078 + ctheodoris/Geneformer V2-104M")
    main()
