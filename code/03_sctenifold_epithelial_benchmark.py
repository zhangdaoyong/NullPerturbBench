#!/usr/bin/env python3
"""Stage 3: Null-controlled
scTenifoldKnk single-gene and combinatorial virtual knockouts in patient-balanced ER+
malignant breast epithelial cells (Wu et al. GSE176078), benchmarked head-to-head
against Geneformer-V2-104M.

scTenifoldKnk workflow (Osorio et al. 2020, 2022):
  1. makeNetworks: 10 random cell subsamples (n=500 cells each) -> exact directed pcNet
     (nPC=3 principal component regression, scaled to [-1, 1], top 5% edges retained, q=0.95)
  2. tensorDecomposition: CANDECOMP/PARAFAC (CP-ALS, rank=5) tensor denoising across the
     (G x G x 10) directed GRN stack -> consensus denoised directed GRN W_WT
  3. Directed virtual knockout: zero outgoing row(s) only (W_KO[t_pos, :] = 0)
  4. manifoldAlignment: joint 2G x 2G directed graph Laplacian L = D - W over shifted
     positive affinities W1 = W_WT + 1, W2 = W_KO + 1, coupling W12 = mu * I with
     mu = 0.9 * (sum(W1) + sum(W2)) / (2G), extracting d=2 non-trivial real eigenvectors
     (plus d=30 sensitivity check) and Euclidean displacement ||Y_WT[g] - Y_KO[g]||_2
  5. dRegulation: Box-Cox / chi-square FC = d^2 / mean(d^2) (df=1) statistics
  6. 100 out-degree- and target-size-matched random KO nulls per perturbation (1,000 runs)

Outputs under results/stage3_sctenifold/:
  - st_grn_denoised.npy, st_gene_names.npy
  - st_<pert_name>_gene_shifts.csv
  - st_null_summary.csv
  - st_perturbation_summary.json
"""
import json
import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import mmread
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigs
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WU = os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq")
GF_DIR = os.path.join(ROOT, "results/stage2_geneformer")
OUT = os.path.join(ROOT, "results/stage3_sctenifold")
os.makedirs(OUT, exist_ok=True)

SEED = 20261119
N_CELLS = 1000
N_GENES = 2000
N_NET = 10
N_SUB_CELLS = 500
NPC = 3
Q_EDGE = 0.95
CP_RANK = 5
CP_ITER = 25
MA_DIM = 2
N_PERM = 100

APP4 = ["HLA-A", "B2M", "TAP1", "PSMB8"]
APP13 = [
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DRB5", "HLA-DPA1",
    "HLA-DPB1", "CD74", "PSMB9", "PSMB8", "TAP1", "TAP2", "B2M"
]
HK_GENES = ["ACTB", "GAPDH", "RPLP0", "HPRT1"]
FORCED_GENES = sorted(set(APP13 + HK_GENES + [
    "HDAC1", "HDAC2", "HDAC3", "HDAC6", "ESR1", "FOXA1", "GATA3",
    "JAK1", "JAK2", "STAT1", "IRF1", "NLRC5", "CIITA"
]))


def build_pcnet_exact(E, nPC=NPC):
    """Exact directed principal-component regression GRN via Gram downdate (machine-precision
    equivalent to per-gene SVD pcNet; max abs diff < 1e-8), scaled to [-1, 1] without global
    edge truncation so lower-detection regulators (HDAC3, JAK2) remain connected."""
    n_cells, n_genes = E.shape
    Ec = (E - E.mean(axis=0)) / (E.std(axis=0) + 1e-8)
    G_all = (Ec @ Ec.T).astype(np.float64)
    W = np.zeros((n_genes, n_genes), dtype=np.float32)
    for g in range(n_genes):
        eg = Ec[:, g].astype(np.float64)
        Gg = G_all - np.outer(eg, eg)
        evals, U = eigh(Gg, subset_by_index=[n_cells - nPC, n_cells - 1])
        evals = np.maximum(evals, 1e-12)
        w_vec = U @ ((U.T @ eg) / evals)
        beta = Ec.T @ w_vec
        beta[g] = 0.0
        W[g, :] = beta.astype(np.float32)
    mx = float(np.max(np.abs(W)))
    if mx > 0:
        W /= mx
    return W


def cp_als_tensor_denoise(net_list, rank=CP_RANK, n_iter=CP_ITER, seed=SEED):
    """CANDECOMP/PARAFAC (CP-ALS) tensor denoising across subsampled directed GRNs."""
    rng_c = np.random.default_rng(seed)
    T = np.stack(net_list, axis=2).astype(np.float64)  # G x G x N_NET
    G, _, K = T.shape
    A = rng_c.standard_normal((G, rank))
    B = rng_c.standard_normal((G, rank))
    C = rng_c.standard_normal((K, rank))
    A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    eye_r = 1e-8 * np.eye(rank)
    for _ in range(n_iter):
        M1 = np.einsum("ijk,jr,kr->ir", T, B, C)
        A = np.linalg.solve((B.T @ B) * (C.T @ C) + eye_r, M1.T).T
        A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)

        M2 = np.einsum("ijk,ir,kr->jr", T, A, C)
        B = np.linalg.solve((A.T @ A) * (C.T @ C) + eye_r, M2.T).T
        B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)

        M3 = np.einsum("ijk,ir,jr->kr", T, A, B)
        C = np.linalg.solve((A.T @ A) * (B.T @ B) + eye_r, M3.T).T

    # Consensus denoised directed GRN averaged across subnetwork slices
    c_mean = C.mean(axis=0)
    W_hat = (A * c_mean[None, :]) @ B.T
    np.fill_diagonal(W_hat, 0.0)
    mx = float(np.max(np.abs(W_hat)))
    if mx > 0:
        W_hat /= mx
    return W_hat


def manifold_alignment_ko(W_wt, ko_positions, d_dim=MA_DIM):
    """scTenifoldKnk directed virtual KO (zero outgoing row(s) only) and joint
    directed graph Laplacian manifold alignment over non-negative edge magnitudes |W|."""
    G = W_wt.shape[0]
    W_ko = W_wt.copy()
    for gp in ko_positions:
        W_ko[gp, :] = 0.0  # scTenifoldKnk: zero outgoing row(s) only

    W1 = np.abs(W_wt).astype(np.float64)
    W2 = np.abs(W_ko).astype(np.float64)
    np.fill_diagonal(W1, 0.0)
    np.fill_diagonal(W2, 0.0)

    mu = 0.9 * (float(W1.sum()) + float(W2.sum())) / (2.0 * G)
    d1 = W1.sum(axis=1) + mu
    d2 = W2.sum(axis=1) + mu

    def matvec(v):
        v1 = v[:G]
        v2 = v[G:]
        return np.concatenate([
            d1 * v1 - (W1 @ v1) - mu * v2,
            d2 * v2 - (W2 @ v2) - mu * v1,
        ])

    op = LinearOperator((2 * G, 2 * G), matvec=matvec, dtype=np.float64)
    vals, vecs = eigs(op, k=d_dim + 2, which="SR", tol=1e-11, maxiter=1500)
    order = np.argsort(vals.real)
    vecs = vecs[:, order].real
    # Skip trivial constant eigenvector (eigenvalue ~ 0)
    Y = vecs[:, 1: 1 + d_dim]
    return np.linalg.norm(Y[:G] - Y[G:], axis=1)


def d_regulation_stats(dist, keep_mask):
    """scTenifoldKnk dRegulation FC = d^2 / mean(d^2) and chi-square (df=1) p-values."""
    d_sub = dist[keep_mask]
    d2 = d_sub ** 2
    fc = d2 / max(float(np.mean(d2)), 1e-30)
    pvals = stats.chi2.sf(fc, df=1)
    qvals = multipletests(pvals, method="fdr_bh")[1]
    return fc, pvals, qvals


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    meta = pd.read_csv(os.path.join(WU, "metadata.csv"), index_col=0)
    genes = [l.strip() for l in open(os.path.join(WU, "count_matrix_genes.tsv"))]
    barcodes = [l.strip() for l in open(os.path.join(WU, "count_matrix_barcodes.tsv"))]
    X = mmread(os.path.join(WU, "count_matrix_sparse.mtx")).T.tocsr()

    # Patient-balanced 1,000 ER+ Cancer Epithelial cells (all 450 Geneformer balanced core cells + 550 balanced extra)
    core_df = pd.read_csv(os.path.join(GF_DIR, "er_core_pool_cells.csv"))
    core_bc = set(core_df["barcode"])
    er_all_idx = [i for i, b in enumerate(barcodes)
                  if b in meta.index and meta.loc[b, "celltype_major"] == "Cancer Epithelial"
                  and meta.loc[b, "subtype"] == "ER+"]
    core_idx = [i for i in er_all_idx if barcodes[i] in core_bc]
    other_idx = [i for i in er_all_idx if barcodes[i] not in core_bc]

    # Sample remaining cells balanced across ER+ patients
    need = max(0, N_CELLS - len(core_idx))
    by_pat = {}
    for i in other_idx:
        p = meta.loc[barcodes[i], "orig.ident"]
        by_pat.setdefault(p, []).append(i)
    for p in by_pat:
        rng.shuffle(by_pat[p])
    extra_idx = []
    pos = {p: 0 for p in sorted(by_pat.keys())}
    while len(extra_idx) < need:
        progressed = False
        for p in sorted(by_pat.keys()):
            if pos[p] < len(by_pat[p]) and len(extra_idx) < need:
                extra_idx.append(by_pat[p][pos[p]])
                pos[p] += 1
                progressed = True
        if not progressed:
            break
    sel_idx = sorted(core_idx + extra_idx)
    sel_pats = pd.Series([meta.loc[barcodes[i], "orig.ident"] for i in sel_idx]).value_counts()
    print(f"Patient-balanced ER+ GRN sample: {len(sel_idx)} ({len(core_idx)} core + {len(extra_idx)} extra across {len(sel_pats)} patients, max share={sel_pats.max()/len(sel_idx)*100:.1f}%)")

    Xsub = X[sel_idx, :].T  # genes x cells
    lib_all = np.asarray(X[sel_idx, :].sum(axis=1)).ravel().astype(np.float32)
    lib_all[lib_all == 0] = 1.0
    det = np.asarray((Xsub > 0).mean(axis=1)).ravel()
    cand = np.where(det >= 0.05)[0]
    g2i = {g: i for i, g in enumerate(genes)}
    forced_idx = {g2i[g] for g in FORCED_GENES if g in g2i}
    var_needed = N_GENES - len(forced_idx)

    sub_dense = Xsub[cand, :].toarray().astype(np.float32)
    logs = np.log1p(sub_dense / lib_all[None, :] * 1e4)
    top = cand[np.argsort(-logs.var(axis=1))[:var_needed]]
    gene_idx = np.array(sorted(set(top.tolist()) | forced_idx))
    gene_names = [genes[i] for i in gene_idx]
    gpos = {g: i for i, g in enumerate(gene_names)}
    print(f"Selected {len(gene_names)} network genes; building {N_NET} subsampled pcNets (n={N_SUB_CELLS}, nPC={NPC})...")

    block = Xsub[gene_idx, :].toarray().astype(np.float32)
    E_full = np.log1p(block / lib_all[None, :] * 1e4).T  # 1000 x G

    nets = []
    for k in range(N_NET):
        tk = time.time()
        sub_c = rng.choice(len(sel_idx), size=N_SUB_CELLS, replace=False)
        nets.append(build_pcnet_exact(E_full[sub_c], nPC=NPC))
        print(f"  pcNet {k+1}/{N_NET} built in {time.time()-tk:.1f}s")

    tcp = time.time()
    W_wt = cp_als_tensor_denoise(nets, rank=CP_RANK, n_iter=CP_ITER, seed=SEED)
    np.save(os.path.join(OUT, "st_grn_denoised.npy"), W_wt.astype(np.float32))
    np.save(os.path.join(OUT, "st_gene_names.npy"), np.array(gene_names))
    print(f"CP-ALS (rank={CP_RANK}) tensor denoising complete in {time.time()-tcp:.1f}s | total elapsed {time.time()-t0:.1f}s")

    # Out-degree by sum of |W_wt| along rows (matches outgoing-row KO)
    out_deg = np.abs(W_wt).sum(axis=1)
    app4_idx = [gpos[g] for g in APP4 if g in gpos]

    perts = [
        ("del_HDAC1", ["HDAC1"]),
        ("del_HDAC2", ["HDAC2"]),
        ("del_HDAC3", ["HDAC3"]),
        ("del_HDAC1_HDAC2", ["HDAC1", "HDAC2"]),
        ("del_HDAC1_HDAC2_HDAC3", ["HDAC1", "HDAC2", "HDAC3"]),
        ("del_JAK1", ["JAK1"]),
        ("del_JAK2", ["JAK2"]),
        ("del_JAK1_JAK2", ["JAK1", "JAK2"]),
        ("del_STAT1", ["STAT1"]),
        ("del_ESR1", ["ESR1"]),
    ]

    results = {}
    for pname, targets in perts:
        t_pos = [gpos[g] for g in targets if g in gpos]
        d = manifold_alignment_ko(W_wt, t_pos, d_dim=MA_DIM)
        d30 = manifold_alignment_ko(W_wt, t_pos, d_dim=30)

        keep_mask = np.ones(len(gene_names), dtype=bool)
        for tp in t_pos:
            keep_mask[tp] = False
        fc, p_chi, q_chi = d_regulation_stats(d, keep_mask)
        ko_ranks = [int((d >= d[tp]).sum()) for tp in t_pos]

        df = pd.DataFrame({
            "gene": np.array(gene_names)[keep_mask],
            "displacement": d[keep_mask],
            "displacement_d30": d30[keep_mask],
            "FC": fc,
            "p_chi2": p_chi,
            "q_chi2": q_chi,
            "is_APP4": [gene_names[i] in APP4 for i in np.where(keep_mask)[0]],
            "is_APP13": [gene_names[i] in APP13 for i in np.where(keep_mask)[0]],
        }).sort_values("displacement", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df["percentile"] = df["displacement"].rank(pct=True)
        df["percentile_d30"] = df["displacement_d30"].rank(pct=True)
        df.to_csv(os.path.join(OUT, f"st_{pname}_gene_shifts.csv"), index=False)

        gmap = df.set_index("gene")
        a4_vals = [float(gmap.loc[g, "displacement"]) for g in APP4 if g in gmap.index]
        a4_pcts = [float(gmap.loc[g, "percentile"]) for g in APP4 if g in gmap.index]
        a4_pcts_d30 = [float(gmap.loc[g, "percentile_d30"]) for g in APP4 if g in gmap.index]
        a13_vals = [float(gmap.loc[g, "displacement"]) for g in APP13 if g in gmap.index]
        a13_pcts = [float(gmap.loc[g, "percentile"]) for g in APP13 if g in gmap.index]
        hk_vals = [float(gmap.loc[g, "displacement"]) for g in HK_GENES if g in gmap.index]

        top100_set = set(df.head(100)["gene"])
        app13_hits = len(top100_set & set(APP13))
        app13_hyper_p = float(stats.hypergeom.sf(app13_hits - 1, len(df), len(set(APP13) & set(df["gene"])), 100)) if app13_hits > 0 else 1.0
        n_sig_fdr = int((df["q_chi2"] < 0.05).sum())

        # Cross-method correlation with Geneformer (updated if Stage 2 files exist)
        gf_file = os.path.join(GF_DIR, f"gf_{pname}_gene_shifts.csv")
        rho_gf, p_gf, jacc_gf = np.nan, np.nan, np.nan
        if os.path.exists(gf_file):
            gf_df = pd.read_csv(gf_file)
            merged = df[["gene", "displacement"]].merge(gf_df[["gene", "mean_shift"]], on="gene")
            rho_gf, p_gf = stats.spearmanr(merged["displacement"], merged["mean_shift"])
            top100_gf = set(gf_df.nlargest(100, "mean_shift")["gene"])
            jacc_gf = len(top100_set & top100_gf) / max(1, len(top100_set | top100_gf))

        results[pname] = {
            "targets": targets,
            "target_ko_self_ranks": ko_ranks,
            "target_abs_out_degrees": [float(out_deg[gpos[g]]) for g in targets if g in gpos],
            "app4_mean_disp": float(np.mean(a4_vals)),
            "app4_mean_pct": float(np.mean(a4_pcts)),
            "app4_mean_pct_d30": float(np.mean(a4_pcts_d30)),
            "app13_mean_disp": float(np.mean(a13_vals)),
            "app13_mean_pct": float(np.mean(a13_pcts)),
            "hk_mean_disp": float(np.mean(hk_vals)),
            "per_gene_app4_pct": {g: float(gmap.loc[g, "percentile"]) for g in APP4 if g in gmap.index},
            "app13_top100_hits": app13_hits,
            "app13_hyper_p": app13_hyper_p,
            "n_dreg_fdr05": n_sig_fdr,
            "spearman_vs_geneformer": float(rho_gf) if not np.isnan(rho_gf) else None,
            "spearman_p_vs_geneformer": float(p_gf) if not np.isnan(p_gf) else None,
            "top100_jaccard_vs_geneformer": float(jacc_gf) if not np.isnan(jacc_gf) else None,
        }
        print(
            f"  {pname:24s}: KO_rk={ko_ranks} | APP4_pct={results[pname]['app4_mean_pct']*100:4.1f}% (d30={results[pname]['app4_mean_pct_d30']*100:4.1f}%) "
            f"| APP13_top100={app13_hits} | rho_GF={rho_gf:+.3f}"
        )

    # 100 out-degree- and size-matched random knockouts per perturbation (1,000 total alignments)
    print(f"Running {N_PERM} out-degree- and size-matched random KO nulls per perturbation...")
    bg_pool = [i for i, g in enumerate(gene_names) if g not in set(FORCED_GENES)]
    null_rows = []
    for pname, targets in perts:
        t_pos = [gpos[g] for g in targets if g in gpos]
        t_degs = [out_deg[tp] for tp in t_pos]
        emp_disps, emp_pcts = [], []
        for p in range(N_PERM):
            picks = []
            used = set()
            for td in t_degs:
                cands = [i for i in bg_pool if i not in used and abs(out_deg[i] - td) / (td + 1e-9) < 0.25]
                if not cands:
                    cands = [i for i in bg_pool if i not in used]
                pk = int(rng.choice(cands))
                picks.append(pk)
                used.add(pk)
            d_n = manifold_alignment_ko(W_wt, picks, d_dim=MA_DIM)
            keep_n = np.ones(len(gene_names), dtype=bool)
            for pk in picks:
                keep_n[pk] = False
            ranks_full = np.full(len(gene_names), np.nan)
            ranks_full[keep_n] = pd.Series(d_n[keep_n]).rank(pct=True).values
            m_disp = float(np.mean(d_n[app4_idx]))
            m_pct = float(np.nanmean(ranks_full[app4_idx]))
            emp_disps.append(m_disp)
            emp_pcts.append(m_pct)
            null_rows.append({"pert": pname, "perm": p, "app4_mean_disp": m_disp, "app4_mean_pct": m_pct})

        arr_d = np.array(emp_disps)
        arr_p = np.array(emp_pcts)
        obs_d = results[pname]["app4_mean_disp"]
        obs_p = results[pname]["app4_mean_pct"]
        results[pname]["null_app4_disp_mean"] = float(arr_d.mean())
        results[pname]["null_app4_pct_mean"] = float(arr_p.mean())
        results[pname]["null_p_app4_disp"] = float((1 + (arr_d >= obs_d).sum()) / (1 + len(arr_d)))
        results[pname]["null_p_app4_pct"] = float((1 + (arr_p >= obs_p).sum()) / (1 + len(arr_p)))
        print(f"  nulls {pname:24s}: obs_pct={obs_p*100:4.1f}% vs null={arr_p.mean()*100:4.1f}% (P={results[pname]['null_p_app4_pct']:.3f}) [{time.time()-t0:.0f}s]")

    pd.DataFrame(null_rows).to_csv(os.path.join(OUT, "st_null_summary.csv"), index=False)
    results["_meta"] = {
        "algorithm": "scTenifoldKnk (Osorio et al. 2020, 2022): 10-subsample dense pcNet (n=500, nPC=3) + CP-ALS tensor denoising (rank=5) + outgoing-row KO + joint directed graph Laplacian manifold alignment over |W| (d=2, mu=0.9, tol=1e-11) + dRegulation chi2",
        "n_cells": len(sel_idx),
        "n_patients": int(len(sel_pats)),
        "max_patient_share": float(sel_pats.max() / len(sel_idx)),
        "n_genes": len(gene_names),
        "n_perm": N_PERM,
        "p_min_floor": float(1.0 / (1 + N_PERM)),
    }
    json.dump(results, open(os.path.join(OUT, "st_perturbation_summary.json"), "w"), indent=2)
    print(f"=== Stage 3 (scTenifoldKnk) Complete in {time.time()-t0:.1f}s ===")



def _require_path(path: str, dataset_hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(
            f"[DATA REQUIRED] Expected raw input not found: {path}\n"
            f"  Dataset: {dataset_hint}\n"
            f"  See README.md ('Pipeline Reproduction') for public accession download paths."
        )

if __name__ == "__main__":
    _require_path(os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"), "Wu et al. GSE176078")
    main()
