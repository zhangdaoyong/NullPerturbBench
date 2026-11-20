#!/usr/bin/env python3
"""Stage 1: Tumor-intrinsic MHC-I APP suppression across independent scRNA-seq
atlases (Wu GSE176078 & Pal GSE161529) and multi-omic TCGA-BRCA / SCAN-B validation.

Outputs under results/stage1/:
  - wu_epithelial_pseudobulk.csv
  - wu_epithelial_stats.json
  - pal_epithelial_pseudobulk.csv
  - pal_epithelial_stats.json
  - tcga_app_multiomics.csv
  - tcga_app_methylation.csv
  - tcga_scanb_app_stats.json
"""
import gzip
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import mmread
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results/stage1")
os.makedirs(OUT, exist_ok=True)

TARGET_GENES = [
    "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9",
    "NLRC5", "CIITA", "STAT1", "IRF1", "ESR1", "FOXA1", "GATA3",
    "HDAC1", "HDAC2", "HDAC3", "HDAC6", "JAK1", "JAK2"
]
APP4 = ["HLA-A", "B2M", "TAP1", "PSMB8"]
APP8 = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9"]


def analyze_wu_epithelial():
    print("=== 1. Analyzing Wu et al. (GSE176078) Epithelial Compartments ===")
    wu_dir = os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq")
    meta = pd.read_csv(os.path.join(wu_dir, "metadata.csv"), index_col=0)
    genes = [l.strip() for l in open(os.path.join(wu_dir, "count_matrix_genes.tsv"))]
    barcodes = [l.strip() for l in open(os.path.join(wu_dir, "count_matrix_barcodes.tsv"))]
    X = mmread(os.path.join(wu_dir, "count_matrix_sparse.mtx")).T.tocsr()

    g2i = {g: i for i, g in enumerate(genes)}
    avail = [g for g in TARGET_GENES if g in g2i]
    cols = [g2i[g] for g in avail]

    is_epi = meta["celltype_major"].isin(["Cancer Epithelial", "Normal Epithelial"])
    keep_bc = [i for i, b in enumerate(barcodes) if b in meta.index and bool(is_epi.loc[b])]
    bc_names = [barcodes[i] for i in keep_bc]

    X_sub = X[keep_bc, :]
    lib = np.asarray(X_sub.sum(axis=1)).ravel().astype(np.float64)
    lib[lib == 0] = 1.0
    mat = np.log1p(np.asarray(X_sub[:, cols].todense(), dtype=np.float64) / lib[:, None] * 1e4)

    df = pd.DataFrame(mat, index=bc_names, columns=avail)
    df["patient"] = meta.loc[bc_names, "orig.ident"].values
    df["subtype"] = meta.loc[bc_names, "subtype"].values
    df["celltype"] = meta.loc[bc_names, "celltype_major"].values
    df["celltype_minor"] = meta.loc[bc_names, "celltype_minor"].values
    df["APP4"] = df[APP4].mean(axis=1)
    df["APP8"] = df[APP8].mean(axis=1)

    pb = df.groupby(["patient", "subtype", "celltype"])[avail + ["APP4", "APP8"]].mean().reset_index()
    n_cells = df.groupby(["patient", "subtype", "celltype"]).size().reset_index(name="n_cells")
    pb = pb.merge(n_cells, on=["patient", "subtype", "celltype"])
    pb.to_csv(os.path.join(OUT, "wu_epithelial_pseudobulk.csv"), index=False)

    # Filter patients with >= 20 cells per compartment for stable pseudobulk stats
    pb_filt = pb[pb["n_cells"] >= 20].copy()
    norm_pb = pb_filt[pb_filt["celltype"] == "Normal Epithelial"]
    er_pb = pb_filt[(pb_filt["celltype"] == "Cancer Epithelial") & (pb_filt["subtype"] == "ER+")]
    tn_pb = pb_filt[(pb_filt["celltype"] == "Cancer Epithelial") & (pb_filt["subtype"] == "TNBC")]
    her2_pb = pb_filt[(pb_filt["celltype"] == "Cancer Epithelial") & (pb_filt["subtype"] == "HER2+")]

    tested_genes = avail + ["APP4", "APP8"]
    comparisons = {}
    p_ne_list, p_et_list = [], []
    for g in tested_genes:
        u_ne, p_ne = stats.mannwhitneyu(norm_pb[g], er_pb[g], alternative="two-sided")
        u_et, p_et = stats.mannwhitneyu(er_pb[g], tn_pb[g], alternative="two-sided")
        p_ne_list.append(float(p_ne))
        p_et_list.append(float(p_et))
        comparisons[g] = {
            "normal_mean": float(norm_pb[g].mean()),
            "er_cancer_mean": float(er_pb[g].mean()),
            "her2_cancer_mean": float(her2_pb[g].mean()),
            "tnbc_cancer_mean": float(tn_pb[g].mean()),
            "p_normal_vs_er": float(p_ne),
            "p_er_vs_tnbc": float(p_et),
        }
    q_ne = multipletests(p_ne_list, method="fdr_bh")[1]
    q_et = multipletests(p_et_list, method="fdr_bh")[1]
    for g, q1, q2 in zip(tested_genes, q_ne, q_et):
        comparisons[g]["q_normal_vs_er"] = float(q1)
        comparisons[g]["q_er_vs_tnbc"] = float(q2)

    # Normal Epithelial lineage-stratified sensitivity analysis (Mature Luminal / Luminal Progenitors / Myoepithelial)
    normal_subtype_sensitivity = {}
    for minor_lbl in ["Mature Luminal", "Luminal Progenitors", "Myoepithelial"]:
        sub_df = df[(df["celltype"] == "Normal Epithelial") & (df["celltype_minor"] == minor_lbl)]
        sub_pb = sub_df.groupby("patient")[avail + ["APP4", "APP8"]].mean().reset_index()
        sub_n = sub_df.groupby("patient").size().reset_index(name="n_cells")
        sub_pb = sub_pb.merge(sub_n, on="patient")
        sub_pb = sub_pb[sub_pb["n_cells"] >= 20]
        rec_sub = {"n_patients_ge20": int(len(sub_pb)), "n_cells_total": int(len(sub_df)), "genes": {}}
        for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "ESR1", "HDAC1", "HDAC2", "HDAC3"]:
            _, p_sub = stats.mannwhitneyu(sub_pb[g], er_pb[g], alternative="two-sided")
            rec_sub["genes"][g] = {
                "normal_subtype_mean": float(sub_pb[g].mean()),
                "er_cancer_mean": float(er_pb[g].mean()),
                "p_vs_er": float(p_sub),
            }
        normal_subtype_sensitivity[minor_lbl] = rec_sub

    # Cell-level Spearman correlations inside ER+ Cancer Epithelial (n=11,878)
    er_cells = df[(df["celltype"] == "Cancer Epithelial") & (df["subtype"] == "ER+")]
    er_corrs = {}
    for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "STAT1"]:
        r_esr1, p_esr1 = stats.spearmanr(er_cells["ESR1"], er_cells[g])
        r_hdac2, p_hdac2 = stats.spearmanr(er_cells["HDAC2"], er_cells[g])
        er_corrs[g] = {
            "rho_ESR1": float(r_esr1), "p_ESR1": float(p_esr1),
            "rho_HDAC2": float(r_hdac2), "p_HDAC2": float(p_hdac2),
        }

    # Pseudobulk correlation across all Cancer Epithelial patients (n=20)
    cancer_pb = pb_filt[pb_filt["celltype"] == "Cancer Epithelial"]
    pb_corrs = {}
    for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "HDAC2"]:
        r_e, p_e = stats.spearmanr(cancer_pb["ESR1"], cancer_pb[g])
        pb_corrs[g] = {"rho_ESR1_pb": float(r_e), "p_ESR1_pb": float(p_e)}

    summary = {
        "n_total_epithelial_cells": int(len(df)),
        "n_er_cancer_cells": int(len(er_cells)),
        "n_patients_normal_ge20": int(len(norm_pb)),
        "n_patients_er_ge20": int(len(er_pb)),
        "n_patients_tnbc_ge20": int(len(tn_pb)),
        "n_patients_her2_ge20": int(len(her2_pb)),
        "comparisons": comparisons,
        "normal_subtype_sensitivity": normal_subtype_sensitivity,
        "er_cell_correlations": er_corrs,
        "cancer_pb_correlations": pb_corrs,
    }
    json.dump(summary, open(os.path.join(OUT, "wu_epithelial_stats.json"), "w"), indent=2)
    print(f"Wu Epithelial: {len(df)} cells | ER+ (n={len(er_pb)}) vs TNBC (n={len(tn_pb)}) APP4 p={comparisons['APP4']['p_er_vs_tnbc']:.4g}, NLRC5 p={comparisons['NLRC5']['p_er_vs_tnbc']:.4g}")


def analyze_pal_epithelial():
    print("=== 2. Analyzing Pal et al. (GSE161529) Independent Epithelial Cohort ===")
    pal_dir = os.path.join(ROOT, "data/processed/scrna_bassez")
    feats = pd.read_csv(os.path.join(ROOT, "data/raw/scrna/GSE161529_features.tsv.gz"), sep="\t", header=None)
    feats.columns = ["id", "symbol", "type"][:feats.shape[1]]
    sym_to_idx = {}
    for idx, sym in enumerate(feats["symbol"]):
        if sym not in sym_to_idx:
            sym_to_idx[sym] = idx

    epi_markers = [g for g in ["EPCAM", "KRT18", "KRT19"] if g in sym_to_idx]
    lum_markers = [g for g in ["KRT18", "KRT19"] if g in sym_to_idx]
    myo_markers = [g for g in ["KRT5", "KRT14", "ACTA2"] if g in sym_to_idx]
    imm_markers = [g for g in ["PTPRC", "CD3D", "CD68"] if g in sym_to_idx]
    avail = [g for g in TARGET_GENES if g in sym_to_idx]
    all_use = list(dict.fromkeys(avail + epi_markers + myo_markers + imm_markers))
    idx_list = [sym_to_idx[g] for g in all_use]

    files = sorted(f for f in os.listdir(pal_dir) if f.endswith("-matrix.mtx.gz"))
    rows, rows_lum = [], []
    for fname in files:
        sample = re.sub(r"^GSM\d+_", "", fname.replace("-matrix.mtx.gz", ""))
        # Exclude lymph node metastases (-LN) and male ER (mER) to keep primary female cohort clean
        if "-LN" in sample or sample.startswith("mER"):
            continue
        m = re.match(r"^(N|B1|TN|HER2|ER)-(.+)$", sample)
        if not m:
            continue
        grp = m.group(1)
        if grp == "B1":
            continue  # BRCA1 preneoplastic excluded from primary subtype comparison
        group_label = "Normal" if grp == "N" else ("ER+" if grp == "ER" else ("HER2+" if grp == "HER2" else "TNBC"))

        X = mmread(os.path.join(pal_dir, fname)).T.tocsr()
        lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        keep_lib = lib >= 500
        if keep_lib.sum() < 20:
            continue
        X = X[keep_lib, :]
        lib = lib[keep_lib]

        sub_counts = np.asarray(X[:, idx_list].todense(), dtype=np.float64)
        df_c = pd.DataFrame(sub_counts, columns=all_use)

        # Isolate pure epithelial cells: express EPCAM or KRT18 or KRT19, and ZERO PTPRC/CD3D/CD68
        epi_sum = df_c[epi_markers].sum(axis=1)
        imm_sum = df_c[imm_markers].sum(axis=1)
        epi_mask = (epi_sum > 0) & (imm_sum == 0)
        n_epi = int(epi_mask.sum())
        if n_epi < 50:
            continue

        logn = np.log1p(sub_counts[epi_mask.values, :] / lib[epi_mask.values, None] * 1e4)
        df_epi = pd.DataFrame(logn[:, :len(avail)], columns=avail)
        df_epi["APP4"] = df_epi[APP4].mean(axis=1)
        df_epi["APP8"] = df_epi[APP8].mean(axis=1)
        means = df_epi.mean().to_dict()
        means.update({"sample": sample, "group": group_label, "n_total_qc": int(len(lib)), "n_epithelial": n_epi})
        rows.append(means)

        # Strict luminal-restricted sensitivity gate: KRT18/19 > 0, KRT5/14/ACTA2 == 0, immune == 0
        lum_mask = (df_c[lum_markers].sum(axis=1) > 0) & (df_c[myo_markers].sum(axis=1) == 0) & (imm_sum == 0)
        n_lum = int(lum_mask.sum())
        if n_lum >= 50:
            logn_l = np.log1p(sub_counts[lum_mask.values, :] / lib[lum_mask.values, None] * 1e4)
            df_l = pd.DataFrame(logn_l[:, :len(avail)], columns=avail)
            df_l["APP4"] = df_l[APP4].mean(axis=1)
            df_l["APP8"] = df_l[APP8].mean(axis=1)
            m_l = df_l.mean().to_dict()
            m_l.update({"sample": sample, "group": group_label, "n_luminal": n_lum})
            rows_lum.append(m_l)

    pb = pd.DataFrame(rows)
    pb.to_csv(os.path.join(OUT, "pal_epithelial_pseudobulk.csv"), index=False)

    norm_df = pb[pb["group"] == "Normal"]
    er_df = pb[pb["group"] == "ER+"]
    tn_df = pb[pb["group"] == "TNBC"]
    her2_df = pb[pb["group"] == "HER2+"]

    tested_pal = avail + ["APP4", "APP8"]
    comp = {}
    p_ne_l, p_et_l = [], []
    for g in tested_pal:
        _, p_ne = stats.mannwhitneyu(norm_df[g], er_df[g], alternative="two-sided")
        _, p_et = stats.mannwhitneyu(er_df[g], tn_df[g], alternative="two-sided")
        p_ne_l.append(float(p_ne))
        p_et_l.append(float(p_et))
        comp[g] = {
            "normal_mean": float(norm_df[g].mean()),
            "er_mean": float(er_df[g].mean()),
            "her2_mean": float(her2_df[g].mean()) if len(her2_df) else None,
            "tnbc_mean": float(tn_df[g].mean()),
            "p_normal_vs_er": float(p_ne),
            "p_er_vs_tnbc": float(p_et),
        }
    q_ne_p = multipletests(p_ne_l, method="fdr_bh")[1]
    q_et_p = multipletests(p_et_l, method="fdr_bh")[1]
    for g, q1, q2 in zip(tested_pal, q_ne_p, q_et_p):
        comp[g]["q_normal_vs_er"] = float(q1)
        comp[g]["q_er_vs_tnbc"] = float(q2)

    pb_lum = pd.DataFrame(rows_lum)
    lum_sens = {
        "n_samples_normal": int((pb_lum["group"] == "Normal").sum()),
        "n_samples_er": int((pb_lum["group"] == "ER+").sum()),
        "n_samples_tnbc": int((pb_lum["group"] == "TNBC").sum()),
        "n_total_luminal_cells": int(pb_lum["n_luminal"].sum()),
        "genes": {},
    }
    ln_df = pb_lum[pb_lum["group"] == "Normal"]
    le_df = pb_lum[pb_lum["group"] == "ER+"]
    lt_df = pb_lum[pb_lum["group"] == "TNBC"]
    for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "ESR1", "HDAC1", "HDAC2", "HDAC3"]:
        _, p_lne = stats.mannwhitneyu(ln_df[g], le_df[g], alternative="two-sided")
        _, p_let = stats.mannwhitneyu(le_df[g], lt_df[g], alternative="two-sided")
        lum_sens["genes"][g] = {
            "normal_luminal_mean": float(ln_df[g].mean()),
            "er_luminal_mean": float(le_df[g].mean()),
            "tnbc_luminal_mean": float(lt_df[g].mean()),
            "p_normal_vs_er": float(p_lne),
            "p_er_vs_tnbc": float(p_let),
        }

    pal_summary = {
        "n_samples_total": int(len(pb)),
        "n_total_epithelial_cells": int(pb["n_epithelial"].sum()),
        "group_counts": pb["group"].value_counts().to_dict(),
        "comparisons": comp,
        "strict_luminal_sensitivity": lum_sens,
    }
    json.dump(pal_summary, open(os.path.join(OUT, "pal_epithelial_stats.json"), "w"), indent=2)
    print(f"Pal GSE161529 Epithelial: {len(pb)} samples ({pb['n_epithelial'].sum()} pure epithelial cells) | groups={pal_summary['group_counts']}")
    print(f"  APP4: Norm={comp['APP4']['normal_mean']:.3f}, ER+={comp['APP4']['er_mean']:.3f}, TNBC={comp['APP4']['tnbc_mean']:.3f} | P(Norm vs ER+)={comp['APP4']['p_normal_vs_er']:.4g}, P(ER+ vs TNBC)={comp['APP4']['p_er_vs_tnbc']:.4g}")


def analyze_tcga_scanb_multiomics():
    print("=== 3. Analyzing TCGA-BRCA Multi-omics & SCAN-B Subtypes ===")
    expr_t = pd.read_parquet(os.path.join(ROOT, "data/processed/expr_tumor.parquet"))
    expr_n = pd.read_parquet(os.path.join(ROOT, "data/processed/expr_normal.parquet"))
    clin = pd.read_csv(os.path.join(ROOT, "data/processed/clin_cohort.csv"), index_col=0)
    imm = pd.read_parquet(os.path.join(ROOT, "data/processed/immune_tcga.parquet"))

    avail = [g for g in TARGET_GENES if g in expr_t.columns]
    df_t = expr_t[avail].copy()
    df_t["APP4"] = df_t[APP4].mean(axis=1)
    df_t["APP8"] = df_t[APP8].mean(axis=1)
    df_t = df_t.join(clin[["PAM50", "ER", "PR", "HER2", "stage", "OS", "OS.time"]])

    # Add pan-leukocyte PTPRC expression + quanTIseq CD8/CYT for immune-content adjustment
    if "PTPRC" in expr_t.columns:
        df_t["PTPRC"] = expr_t["PTPRC"]
    if "qTIL10_T.cells.CD8" in imm.columns:
        df_t = df_t.join(imm[["qTIL10_T.cells.CD8", "CYT"]].rename(
            columns={"qTIL10_T.cells.CD8": "q_T_cells_CD8", "CYT": "CYT_score"}
        ))

    # Normal breast expression
    df_n = expr_n[avail].copy()
    df_n["APP4"] = df_n[APP4].mean(axis=1)
    df_n["APP8"] = df_n[APP8].mean(axis=1)

    # Check promoter methylation of APP genes + NLRC5 in TCGA 450K
    ann = pd.read_csv(os.path.join(ROOT, "data/raw/TCGA_BRCA_methyl450_probe_annotation.csv"))
    focus_meth_genes = ["HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "ESR1", "HDAC1", "HDAC2"]
    # Identify probes mapped to these genes near promoter TSS if gene column matches
    probe_map = {}
    for g in focus_meth_genes:
        m = ann["genes"].fillna("").str.contains(rf"'{re.escape(g)}'", regex=True)
        probes = set(ann.loc[m, "probe"])
        if probes:
            probe_map[g] = probes

    all_target_probes = set().union(*probe_map.values())
    print(f"Scanning TCGA 450K methylation for {len(all_target_probes)} probes across {len(probe_map)} genes...")
    meth_rows = []
    with gzip.open(os.path.join(ROOT, "data/raw/TCGA_BRCA_methyl450_full.csv.gz"), "rt") as f:
        header = f.readline().strip().split(",")
        samples_m = header[1:]
        for line in f:
            parts = line.rstrip("\n").split(",")
            pr = parts[0]
            if pr in all_target_probes:
                vals = [np.nan if v in ("", "NA", "NaN") else float(v) for v in parts[1:]]
                meth_rows.append((pr, vals))
    meth_df = pd.DataFrame({pr: vals for pr, vals in meth_rows}, index=samples_m)

    # Keep tumor samples in df_t and average beta per gene
    common_m = df_t.index.intersection(meth_df.index)
    meth_gene_df = pd.DataFrame(index=df_t.index)
    meth_stats = {}
    for g, probes in probe_map.items():
        p_avail = [p for p in probes if p in meth_df.columns]
        if not p_avail:
            continue
        # compute correlation with expression across tumors to pick promoter-coupled probes or mean
        g_vals = meth_df.loc[common_m, p_avail].mean(axis=1)
        meth_gene_df.loc[common_m, f"meth_{g}"] = g_vals
        valid = g_vals.dropna().index.intersection(df_t.index)
        rho_expr, p_expr = stats.spearmanr(g_vals.loc[valid], df_t.loc[valid, g])
        lum = df_t.loc[valid, "PAM50"].isin(["Luminal A", "Luminal B"])
        bas = df_t.loc[valid, "PAM50"] == "Basal-like"
        _, p_lb = stats.mannwhitneyu(g_vals.loc[valid][lum], g_vals.loc[valid][bas])
        meth_stats[g] = {
            "n_probes": len(p_avail),
            "mean_beta_luminal": float(g_vals.loc[valid][lum].mean()),
            "mean_beta_basal": float(g_vals.loc[valid][bas].mean()),
            "p_lum_vs_basal_meth": float(p_lb),
            "rho_meth_vs_expr": float(rho_expr),
            "p_meth_vs_expr": float(p_expr),
        }
    df_t = df_t.join(meth_gene_df)

    # GISTIC2 copy-number calls for 6p21 (HLA-A, TAP1, PSMB8), 15q21 (B2M), 16q13 (NLRC5)
    gistic = pd.read_csv(os.path.join(ROOT, "data/raw/TCGA_BRCA_GISTIC_thresholded.csv.gz"), index_col=0)
    cnv_genes = ["HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "HDAC1", "HDAC2", "HDAC3"]
    common_cnv = [s for s in df_t.index if s in gistic.columns]
    cnv_stats = {}
    for g in cnv_genes:
        if g in gistic.index:
            vals = gistic.loc[g, common_cnv]
            df_t.loc[common_cnv, f"cnv_{g}"] = vals.values
            lum = df_t.loc[common_cnv, "PAM50"].isin(["Luminal A", "Luminal B"])
            bas = df_t.loc[common_cnv, "PAM50"] == "Basal-like"
            cnv_stats[g] = {
                "loss_freq_all": float((vals < 0).mean()),
                "loss_freq_luminal": float((vals[lum] < 0).mean()),
                "loss_freq_basal": float((vals[bas] < 0).mean()),
                "gain_freq_luminal": float((vals[lum] > 0).mean()),
                "gain_freq_basal": float((vals[bas] > 0).mean()),
            }

    df_t.to_csv(os.path.join(OUT, "tcga_app_multiomics.csv"))
    pd.DataFrame(meth_stats).T.to_csv(os.path.join(OUT, "tcga_app_methylation.csv"))

    # Partial correlation of ESR1 vs APP4 adjusting for PTPRC + CD8 T-cell infiltrate in TCGA
    tcga_purity_adj = {}
    if "PTPRC" in df_t.columns and "q_T_cells_CD8" in df_t.columns:
        sub_p = df_t[["ESR1", "APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "PTPRC", "q_T_cells_CD8"]].dropna()
        for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5"]:
            r_raw, p_raw = stats.spearmanr(sub_p["ESR1"], sub_p[g])
            X_ols = sm.add_constant(sub_p[["ESR1", "PTPRC", "q_T_cells_CD8"]])
            mod = sm.OLS(sub_p[g], X_ols).fit()
            tcga_purity_adj[g] = {
                "spearman_raw": float(r_raw), "p_raw": float(p_raw),
                "ols_beta_ESR1_adj_immune": float(mod.params["ESR1"]),
                "ols_p_ESR1_adj_immune": float(mod.pvalues["ESR1"]),
            }

    # SCAN-B validation
    scan_df = pd.read_csv(os.path.join(ROOT, "results/validation_SCANB.csv"))
    # also read raw SCANB matrix only for APP4 + ESR1 if needed; check validation_SCANB.csv columns
    print("SCAN-B validation file cols:", list(scan_df.columns)[:15])
    # Let's extract APP4/ESR1/PAM50 directly from GSE96058 if needed in quick read
    scan_expr_path = os.path.join(ROOT, "data/raw/geo/GSE96058_gene_expression_transformed.csv.gz")
    want_scan = set(APP4 + ["NLRC5", "STAT1", "ESR1", "HDAC1", "HDAC2", "HDAC3"])
    scan_rows = {}
    with gzip.open(scan_expr_path, "rt") as f:
        hdr = [x.strip('"') for x in f.readline().strip().split(",")]
        for line in f:
            first_comma = line.find(",")
            gene_sym = line[:first_comma].strip('"')
            if gene_sym in want_scan:
                vals = [float(x) for x in line[first_comma+1:].strip().split(",")]
                scan_rows[gene_sym] = vals
    scan_g = pd.DataFrame(scan_rows, index=hdr[1:])
    scan_g["APP4"] = scan_g[APP4].mean(axis=1)

    # Load SCAN-B PAM50/ER metadata from GSE96058_series_matrix or validation_SCANB.csv
    # Note validation_SCANB.csv has patient/sample index aligned with GSE96058
    scanb_corrs = {}
    for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5", "HDAC2"]:
        r_s, p_s = stats.spearmanr(scan_g["ESR1"], scan_g[g])
        scanb_corrs[g] = {"rho_ESR1": float(r_s), "p_ESR1": float(p_s)}

    out_summary = {
        "tcga_n_tumor": int(len(df_t)),
        "tcga_n_normal": int(len(df_n)),
        "tcga_luminal_vs_basal_APP4": {
            "lumA_mean": float(df_t.loc[df_t["PAM50"]=="Luminal A", "APP4"].mean()),
            "lumB_mean": float(df_t.loc[df_t["PAM50"]=="Luminal B", "APP4"].mean()),
            "basal_mean": float(df_t.loc[df_t["PAM50"]=="Basal-like", "APP4"].mean()),
            "normal_mean": float(df_n["APP4"].mean()),
            "p_lumA_vs_basal": float(stats.mannwhitneyu(
                df_t.loc[df_t["PAM50"]=="Luminal A", "APP4"],
                df_t.loc[df_t["PAM50"]=="Basal-like", "APP4"])[1]),
        },
        "tcga_purity_adjusted_ESR1": tcga_purity_adj,
        "tcga_methylation": meth_stats,
        "tcga_cnv": cnv_stats,
        "scanb_n": int(len(scan_g)),
        "scanb_esr1_correlations": scanb_corrs,
    }
    json.dump(out_summary, open(os.path.join(OUT, "tcga_scanb_app_stats.json"), "w"), indent=2)
    print(f"TCGA APP4 LumA={out_summary['tcga_luminal_vs_basal_APP4']['lumA_mean']:.2f} vs Basal={out_summary['tcga_luminal_vs_basal_APP4']['basal_mean']:.2f} (p={out_summary['tcga_luminal_vs_basal_APP4']['p_lumA_vs_basal']:.2e})")
    print(f"SCAN-B (n={len(scan_g)}) ESR1 vs APP4 rho={scanb_corrs['APP4']['rho_ESR1']:.3f} (p={scanb_corrs['APP4']['p_ESR1']:.2e})")



def _require_path(path: str, dataset_hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(
            f"[DATA REQUIRED] Expected raw input not found: {path}\n"
            f"  Dataset: {dataset_hint}\n"
            f"  See README.md ('Pipeline Reproduction') for public accession download paths."
        )

if __name__ == "__main__":
    _require_path(os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"), "Wu et al. GSE176078, Pal et al. GSE161529, TCGA-BRCA (Xena), SCAN-B")
    analyze_wu_epithelial()
    analyze_pal_epithelial()
    analyze_tcga_scanb_multiomics()
    print("=== Stage 1 Complete ===")
