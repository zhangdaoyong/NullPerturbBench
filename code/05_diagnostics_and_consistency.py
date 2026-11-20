#!/usr/bin/env python3
"""Stage 5: Inductive-bias diagnostics, two-covariate OLS residualization,
TCGA immune-purity regressions, and cross-cohort Meta / Bootstrap 95% CI quantification.

Uses log1p(CP10K) baseline ER+ expression (matching Stage 1) and Geneformer-V2-104M
and scTenifoldKnk perturbation outputs.

Outputs under stage5_diagnostics/:
  - inductive_bias_summary.json
  - del_HDAC2_bias_table.csv
  - cross_method_raw_vs_resid.csv
  - uncertainty_meta_summary.json
  - uncertainty_meta_and_ci.csv
"""
import json
import os
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy import stats
from scipy.io import mmread

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ST1 = os.path.join(ROOT, "results/stage1")
GF_DIR = os.path.join(ROOT, "results/stage2_geneformer")
ST_DIR = os.path.join(ROOT, "results/stage3_sctenifold")
L4_DIR = os.path.join(ROOT, "results/stage4_lincs")
OUT = os.path.join(ROOT, "results/stage5_diagnostics")
os.makedirs(OUT, exist_ok=True)

APP4 = ["HLA-A", "B2M", "TAP1", "PSMB8"]


def finish_stage1_purity():
    df_t = pd.read_csv(os.path.join(ST1, "tcga_app_multiomics.csv"), index_col=0)
    expr_t = pd.read_parquet(os.path.join(ROOT, "data/processed/expr_tumor.parquet"))
    imm = pd.read_parquet(os.path.join(ROOT, "data/processed/immune_tcga.parquet"))
    df_t["PTPRC"] = expr_t.loc[df_t.index, "PTPRC"]
    df_t["q_CD8"] = imm.loc[df_t.index, "qTIL10_T.cells.CD8"]
    df_t["is_luminal"] = df_t["PAM50"].isin(["Luminal A", "Luminal B"]).astype(float)

    adj_stats = {}
    for g in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5"]:
        sub = df_t[["ESR1", "PTPRC", "q_CD8", g]].dropna()
        r_raw, p_raw = stats.spearmanr(sub["ESR1"], sub[g])
        X = sm.add_constant(sub[["ESR1", "PTPRC", "q_CD8"]])
        m = sm.OLS(sub[g], X).fit()
        adj_stats[g] = {
            "spearman_raw": float(r_raw), "p_raw": float(p_raw),
            "ols_beta_ESR1_adj_immune": float(m.params["ESR1"]),
            "ols_t_ESR1_adj_immune": float(m.tvalues["ESR1"]),
            "ols_p_ESR1_adj_immune": float(m.pvalues["ESR1"]),
        }

    meth_adj = {}
    for g in ["HLA-A", "B2M", "TAP1", "PSMB8", "NLRC5"]:
        mcol = f"meth_{g}"
        if mcol in df_t.columns:
            sub = df_t[df_t["PAM50"].isin(["Luminal A", "Luminal B", "Basal-like"])][
                ["is_luminal", "PTPRC", "q_CD8", mcol]
            ].dropna()
            X = sm.add_constant(sub[["is_luminal", "PTPRC", "q_CD8"]])
            m = sm.OLS(sub[mcol], X).fit()
            meth_adj[g] = {
                "beta_luminal_adj_immune": float(m.params["is_luminal"]),
                "p_luminal_adj_immune": float(m.pvalues["is_luminal"]),
            }

    st1_json_path = os.path.join(ST1, "tcga_scanb_app_stats.json")
    s1 = json.load(open(st1_json_path))
    s1["tcga_purity_adjusted_ESR1"] = adj_stats
    s1["tcga_purity_adjusted_methylation"] = meth_adj
    json.dump(s1, open(st1_json_path, "w"), indent=2)
    print("Updated Stage 1 JSON with PTPRC+CD8 adjusted ESR1 & methylation regressions.")


def run_inductive_bias_diagnostics():
    WU = os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq")
    meta = pd.read_csv(os.path.join(WU, "metadata.csv"), index_col=0)
    genes = [l.strip() for l in open(os.path.join(WU, "count_matrix_genes.tsv"))]
    barcodes = [l.strip() for l in open(os.path.join(WU, "count_matrix_barcodes.tsv"))]
    X = mmread(os.path.join(WU, "count_matrix_sparse.mtx")).T.tocsr()
    er_idx = [i for i, b in enumerate(barcodes)
              if b in meta.index and meta.loc[b, "celltype_major"] == "Cancer Epithelial"
              and meta.loc[b, "subtype"] == "ER+"]
    Xer = X[er_idx, :]

    # Exact log1p(CP10K) baseline ER+ epithelial mean expression & detection fraction
    lib_er = np.asarray(Xer.sum(axis=1)).ravel().astype(np.float64)
    lib_er[lib_er == 0] = 1.0
    X_cp10k = sp.diags(1e4 / lib_er) @ Xer
    X_cp10k.data = np.log1p(X_cp10k.data)
    mean_expr = pd.Series(np.asarray(X_cp10k.mean(axis=0)).ravel(), index=genes)
    det_rate = pd.Series(np.asarray((Xer > 0).mean(axis=0)).ravel(), index=genes)

    # Load denoised directed GRN (or fallback) and compute total |degree| & out-|degree|
    grn_path = os.path.join(ST_DIR, "st_grn_denoised.npy")
    if not os.path.exists(grn_path):
        grn_path = os.path.join(ST_DIR, "st_grn_sym.npy")
    W = np.load(grn_path)
    gnames = np.load(os.path.join(ST_DIR, "st_gene_names.npy"), allow_pickle=True)
    out_deg = pd.Series(np.abs(W).sum(axis=1), index=gnames)
    tot_deg = pd.Series(np.abs(W).sum(axis=1) + np.abs(W).sum(axis=0), index=gnames)
    # Primary GRN degree metric: out-degree (matches outgoing-row KO in scTenifoldKnk)
    deg_s = out_deg

    perts = [
        "del_HDAC1", "del_HDAC2", "del_HDAC3", "del_HDAC1_HDAC2", "del_HDAC1_HDAC2_HDAC3",
        "del_JAK1", "del_JAK2", "del_JAK1_JAK2", "del_STAT1", "del_ESR1"
    ]

    diag_rows = []
    for p in perts:
        gf = pd.read_csv(os.path.join(GF_DIR, f"gf_{p}_gene_shifts.csv")).set_index("gene")
        st = pd.read_csv(os.path.join(ST_DIR, f"st_{p}_gene_shifts.csv")).set_index("gene")
        sh = gf.index.intersection(st.index).intersection(deg_s.index)

        gf_vals = gf.loc[sh, "mean_shift"]
        st_vals = st.loc[sh, "displacement"]
        me_vals = mean_expr.loc[sh]
        dr_vals = det_rate.loc[sh]
        dg_vals = deg_s.loc[sh]
        tdg_vals = tot_deg.loc[sh]

        r_gf_expr, _ = stats.spearmanr(gf_vals, me_vals)
        r_gf_det, _ = stats.spearmanr(gf_vals, dr_vals)
        r_st_expr, _ = stats.spearmanr(st_vals, me_vals)
        r_st_deg, _ = stats.spearmanr(st_vals, dg_vals)
        r_st_totdeg, _ = stats.spearmanr(st_vals, tdg_vals)
        r_raw_cross, p_raw_cross = stats.spearmanr(gf_vals, st_vals)

        # Two-covariate OLS residualization (documented directly in Materials & Methods):
        #   GF: log10(gf_shift + 1e-8) ~ log10(mean_expr_cp10k + 1e-4) + det_rate
        #   ST: log10(st_disp + 1e-9)  ~ log10(grn_out_deg + 1e-4) + log10(mean_expr_cp10k + 1e-4)
        y_gf = np.log10(gf_vals.values + 1e-8)
        X_gf = sm.add_constant(np.column_stack([np.log10(me_vals.values + 1e-4), dr_vals.values]))
        gf_resid = sm.OLS(y_gf, X_gf).fit().resid

        y_st = np.log10(st_vals.values + 1e-9)
        X_st = sm.add_constant(np.column_stack([np.log10(dg_vals.values + 1e-4), np.log10(me_vals.values + 1e-4)]))
        st_resid = sm.OLS(y_st, X_st).fit().resid

        r_resid_cross, p_resid_cross = stats.spearmanr(gf_resid, st_resid)

        resid_df = pd.DataFrame({"gf_resid": gf_resid, "st_resid": st_resid}, index=sh)
        resid_df["gf_resid_pct"] = resid_df["gf_resid"].rank(pct=True)
        resid_df["st_resid_pct"] = resid_df["st_resid"].rank(pct=True)
        a4_sh = [g for g in APP4 if g in resid_df.index]
        app4_gf_resid_pct = float(resid_df.loc[a4_sh, "gf_resid_pct"].mean())
        app4_st_resid_pct = float(resid_df.loc[a4_sh, "st_resid_pct"].mean())

        diag_rows.append({
            "pert": p, "n_shared": len(sh),
            "rho_GF_vs_expr": float(r_gf_expr), "rho_GF_vs_det": float(r_gf_det),
            "rho_ST_vs_expr": float(r_st_expr), "rho_ST_vs_deg": float(r_st_deg),
            "rho_ST_vs_totdeg": float(r_st_totdeg),
            "rho_cross_raw": float(r_raw_cross), "p_cross_raw": float(p_raw_cross),
            "rho_cross_resid": float(r_resid_cross), "p_cross_resid": float(p_resid_cross),
            "APP4_gf_resid_pct": app4_gf_resid_pct, "APP4_st_resid_pct": app4_st_resid_pct,
        })

        if p == "del_HDAC2":
            tbl = pd.DataFrame({
                "gene": sh,
                "mean_expr": me_vals.values, "det_rate": dr_vals.values,
                "grn_abs_deg": dg_vals.values, "grn_tot_deg": tdg_vals.values,
                "gf_shift": gf_vals.values, "gf_resid": gf_resid,
                "st_disp": st_vals.values, "st_resid": st_resid,
            })
            tbl.to_csv(os.path.join(OUT, "del_HDAC2_bias_table.csv"), index=False)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(os.path.join(OUT, "cross_method_raw_vs_resid.csv"), index=False)

    # Test if resid_GF / resid_ST aligns better with L1000 CRISPR HDAC1/2/3 |z|
    gi = pd.read_csv(os.path.join(ROOT, "data/raw/mr/gene_info.txt.gz"), sep="\t")
    sig = pd.read_csv(os.path.join(ROOT, "data/raw/mr/sig_info.txt.gz"), sep="\t", low_memory=False)
    id2sym = dict(zip(gi["pr_gene_id"].astype(int), gi["pr_gene_symbol"]))
    xpr = sig[sig["cell_id"].str.startswith("MCF7", na=False) & (sig["pert_type"] == "trt_xpr")]
    resid_crispr = {}
    with h5py.File(os.path.join(ROOT, "data/raw/mr/GSE70138_level5.gctx"), "r") as f:
        row_ids = np.array([int(x) for x in f["/0/META/ROW/id"][:]])
        col_ids = [c.decode() if isinstance(c, bytes) else str(c) for c in f["/0/META/COL/id"][:]]
        col_map = {c: i for i, c in enumerate(col_ids)}
        row_syms = [id2sym.get(g, str(g)) for g in row_ids]
        mat = f["/0/DATA/0/matrix"]
        for tg in ["HDAC1", "HDAC2", "HDAC3"]:
            sids = xpr.loc[xpr["pert_iname"] == tg, "sig_id"].tolist()
            cidx = sorted([col_map[s] for s in sids if s in col_map])
            cr_s = pd.Series(mat[cidx, :].mean(axis=0), index=row_syms).groupby(level=0).mean()

            gf = pd.read_csv(os.path.join(GF_DIR, f"gf_del_{tg}_gene_shifts.csv")).set_index("gene")
            st = pd.read_csv(os.path.join(ST_DIR, f"st_del_{tg}_gene_shifts.csv")).set_index("gene")
            sh_g = gf.index.intersection(cr_s.index).intersection(mean_expr.index)
            y_g = np.log10(gf.loc[sh_g, "mean_shift"].values + 1e-8)
            X_g = sm.add_constant(np.column_stack([np.log10(mean_expr.loc[sh_g].values + 1e-4), det_rate.loc[sh_g].values]))
            r_g = sm.OLS(y_g, X_g).fit().resid
            rho_gf_res, p_gf_res = stats.spearmanr(r_g, cr_s.loc[sh_g].abs())

            sh_s = st.index.intersection(cr_s.index).intersection(deg_s.index)
            y_s = np.log10(st.loc[sh_s, "displacement"].values + 1e-9)
            X_s = sm.add_constant(np.column_stack([np.log10(deg_s.loc[sh_s].values + 1e-4), np.log10(mean_expr.loc[sh_s].values + 1e-4)]))
            r_s = sm.OLS(y_s, X_s).fit().resid
            rho_st_res, p_st_res = stats.spearmanr(r_s, cr_s.loc[sh_s].abs())
            resid_crispr[tg] = {
                "gf_resid_vs_crispr_abs_rho": float(rho_gf_res), "gf_resid_p": float(p_gf_res),
                "st_resid_vs_crispr_abs_rho": float(rho_st_res), "st_resid_p": float(p_st_res),
            }

    json.dump({"diagnostics": diag_rows, "resid_vs_crispr": resid_crispr}, open(os.path.join(OUT, "inductive_bias_summary.json"), "w"), indent=2)
    print("\n=== Cross-Method Inductive Bias Diagnostics ===")
    print(diag_df[["pert", "rho_GF_vs_expr", "rho_ST_vs_deg", "rho_cross_raw", "rho_cross_resid", "APP4_gf_resid_pct", "APP4_st_resid_pct"]].round(4).to_string(index=False))
    print("\n=== Residualized Virtual KO vs Empirical MCF-7 CRISPR |z| ===")
    print(json.dumps(resid_crispr, indent=2))


def _hedges_g(x1, x0):
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    n1, n0 = len(x1), len(x0)
    s1, s0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    d = (np.mean(x1) - np.mean(x0)) / sp
    df = n1 + n0 - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    g = J * d
    var_g = (n1 + n0) / (n1 * n0) + (g ** 2) / (2.0 * (n1 + n0))
    return float(g), float(var_g)


def _meta_dl(effects, variances):
    y = np.asarray(effects, float)
    v = np.asarray(variances, float)
    w = 1.0 / v
    fe = float(np.sum(w * y) / np.sum(w))
    se_fe = float(np.sqrt(1.0 / np.sum(w)))
    Q = float(np.sum(w * (y - fe) ** 2))
    df = len(y) - 1
    p_Q = float(1.0 - stats.chi2.cdf(Q, df)) if df > 0 else 1.0
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if df > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q * 100.0) if Q > 0 and df > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z_re = re / se_re
    p_re = float(2.0 * stats.norm.sf(abs(z_re)))
    return {
        "fe_g": fe, "fe_ci_lo": fe - 1.96 * se_fe, "fe_ci_hi": fe + 1.96 * se_fe,
        "re_g": re, "re_se": se_re, "re_ci_lo": re - 1.96 * se_re, "re_ci_hi": re + 1.96 * se_re,
        "z_re": float(z_re), "p_re": p_re, "Q": Q, "p_Q": p_Q, "I2_pct": float(I2), "tau2": float(tau2),
    }


def _boot_diff(a, b, rng, B=1000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = float(np.mean(a) - np.mean(b))
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, float(lo), float(hi)


def _boot_spearman(x, y, rng, B=1000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    vals = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, size=n)
        vals[k] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi)


def compute_uncertainty_and_meta(seed=20261119, B=1000):
    """Compute DerSimonian-Laird cross-cohort meta-analysis (Wu + Pal) and
    1,000-resample nonparametric bootstrap 95% CIs purely from derived Stage 1-5 tables.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # 1. Cross-cohort Wu + Pal epithelial APP4 meta-analysis & bootstrap mean differences
    wu_pb = pd.read_csv(os.path.join(ST1, "wu_epithelial_pseudobulk.csv"))
    wu_f = wu_pb[wu_pb["n_cells"] >= 20]
    wu_er = wu_f[(wu_f["celltype"] == "Cancer Epithelial") & (wu_f["subtype"] == "ER+")]["APP4"].values
    wu_norm = wu_f[wu_f["celltype"] == "Normal Epithelial"]["APP4"].values
    wu_tn = wu_f[(wu_f["celltype"] == "Cancer Epithelial") & (wu_f["subtype"] == "TNBC")]["APP4"].values

    pal_pb = pd.read_csv(os.path.join(ST1, "pal_epithelial_pseudobulk.csv"))
    pal_er = pal_pb[pal_pb["group"] == "ER+"]["APP4"].values
    pal_norm = pal_pb[pal_pb["group"] == "Normal"]["APP4"].values
    pal_tn = pal_pb[pal_pb["group"] == "TNBC"]["APP4"].values

    g_wu, v_wu = _hedges_g(wu_er, wu_norm)
    g_pal, v_pal = _hedges_g(pal_er, pal_norm)
    meta_norm = _meta_dl([g_wu, g_pal], [v_wu, v_pal])

    g_wu_t, v_wu_t = _hedges_g(wu_er, wu_tn)
    g_pal_t, v_pal_t = _hedges_g(pal_er, pal_tn)
    meta_tnbc = _meta_dl([g_wu_t, g_pal_t], [v_wu_t, v_pal_t])

    wu_d, wu_lo, wu_hi = _boot_diff(wu_er, wu_norm, rng, B)
    pal_d, pal_lo, pal_hi = _boot_diff(pal_er, pal_norm, rng, B)

    rows.extend([
        {"category": "Epithelial_Bootstrap", "comparison": "Wu ER+ vs Normal APP4 mean diff",
         "n_info": f"n={len(wu_er)} vs {len(wu_norm)}", "metric": "Mean diff (log1p CP10K)",
         "estimate": wu_d, "ci_95_low": wu_lo, "ci_95_high": wu_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
        {"category": "Epithelial_Bootstrap", "comparison": "Pal ER+ vs Normal APP4 mean diff",
         "n_info": f"n={len(pal_er)} vs {len(pal_norm)}", "metric": "Mean diff (log1p CP10K)",
         "estimate": pal_d, "ci_95_low": pal_lo, "ci_95_high": pal_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
        {"category": "Cross_Cohort_Meta", "comparison": "Wu + Pal ER+ vs Normal Epithelial APP4 (RE)",
         "n_info": f"n={len(wu_er)+len(pal_er)} vs {len(wu_norm)+len(pal_norm)}", "metric": "Pooled Hedges g (RE)",
         "estimate": meta_norm["re_g"], "ci_95_low": meta_norm["re_ci_lo"], "ci_95_high": meta_norm["re_ci_hi"],
         "p_value": meta_norm["p_re"], "heterogeneity_I2_pct": meta_norm["I2_pct"]},
        {"category": "Cross_Cohort_Meta", "comparison": "Wu + Pal ER+ vs TNBC Epithelial APP4 (RE)",
         "n_info": f"n={len(wu_er)+len(pal_er)} vs {len(wu_tn)+len(pal_tn)}", "metric": "Pooled Hedges g (RE)",
         "estimate": meta_tnbc["re_g"], "ci_95_low": meta_tnbc["re_ci_lo"], "ci_95_high": meta_tnbc["re_ci_hi"],
         "p_value": meta_tnbc["p_re"], "heterogeneity_I2_pct": meta_tnbc["I2_pct"]},
    ])

    # 2. LINCS L1000 Bootstrap 95% CIs
    xpr = pd.read_csv(os.path.join(L4_DIR, "mcf7_crispr_53_app.csv"))
    cp_all = pd.read_csv(os.path.join(L4_DIR, "mcf7_compounds_1760_app.csv"))
    cp_24h = pd.read_csv(os.path.join(L4_DIR, "mcf7_compounds_24h_app.csv"))
    cr_h = xpr[xpr["pert_iname"].isin(["HDAC1", "HDAC2", "HDAC3"])]
    cr_o = xpr[~xpr["pert_iname"].isin(["HDAC1", "HDAC2", "HDAC3"])]
    ca_d, ca_lo, ca_hi = _boot_diff(cr_h["APP4"], cr_o["APP4"], rng, B)
    cp_d, cp_lo, cp_hi = _boot_diff(cr_h["PSMB8"], cr_o["PSMB8"], rng, B)
    da_d, da_lo, da_hi = _boot_diff(cp_all[cp_all["drug_class"] == "HDACi"]["APP4"],
                                    cp_all[cp_all["drug_class"] == "Other"]["APP4"], rng, B)
    d24_d, d24_lo, d24_hi = _boot_diff(cp_24h[cp_24h["drug_class"] == "HDACi"]["APP4"],
                                       cp_24h[cp_24h["drug_class"] == "Other"]["APP4"], rng, B)
    rows.extend([
        {"category": "LINCS_Bootstrap", "comparison": "24 h-matched HDACi vs Other APP4 lift",
         "n_info": "n=8 vs 1,715", "metric": "Mean MODZ diff",
         "estimate": d24_d, "ci_95_low": d24_lo, "ci_95_high": d24_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
        {"category": "LINCS_Bootstrap", "comparison": "All-timepoint HDACi vs Other APP4 lift",
         "n_info": "n=9 vs 1,742", "metric": "Mean MODZ diff",
         "estimate": da_d, "ci_95_low": da_lo, "ci_95_high": da_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
        {"category": "LINCS_Bootstrap", "comparison": "CRISPR HDAC1/2/3 vs 50 KOs APP4 lift",
         "n_info": "n=3 vs 50", "metric": "Mean MODZ diff",
         "estimate": ca_d, "ci_95_low": ca_lo, "ci_95_high": ca_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
        {"category": "LINCS_Bootstrap", "comparison": "CRISPR HDAC1/2/3 vs 50 KOs PSMB8 (LM) lift",
         "n_info": "n=3 vs 50", "metric": "Mean MODZ diff",
         "estimate": cp_d, "ci_95_low": cp_lo, "ci_95_high": cp_hi, "p_value": np.nan, "heterogeneity_I2_pct": np.nan},
    ])

    # 3. Cross-method & inductive-bias Bootstrap 95% CIs
    diag = pd.read_csv(os.path.join(OUT, "cross_method_raw_vs_resid.csv"))
    for p in diag["pert"].tolist():
        gf = pd.read_csv(os.path.join(GF_DIR, f"gf_{p}_gene_shifts.csv")).set_index("gene")
        st = pd.read_csv(os.path.join(ST_DIR, f"st_{p}_gene_shifts.csv")).set_index("gene")
        sh = gf.index.intersection(st.index)
        r_cr, lo_cr, hi_cr = _boot_spearman(gf.loc[sh, "mean_shift"], st.loc[sh, "displacement"], rng, B)
        rows.append({
            "category": "Cross_Method_Bootstrap", "comparison": f"{p} Geneformer-V2 vs scTenifoldKnk (raw)",
            "n_info": f"n={len(sh)} shared genes", "metric": "Spearman rho",
            "estimate": r_cr, "ci_95_low": lo_cr, "ci_95_high": hi_cr, "p_value": np.nan, "heterogeneity_I2_pct": np.nan,
        })

    bias_ref = pd.read_csv(os.path.join(OUT, "del_HDAC2_bias_table.csv")).set_index("gene")
    for label, c1, c2 in [
        ("del_HDAC2 Geneformer-V2 vs baseline mean expr", "gf_shift", "mean_expr"),
        ("del_HDAC2 scTenifoldKnk vs GRN out-degree", "st_disp", "grn_abs_deg"),
        ("del_HDAC2 Geneformer-V2 vs scTenifoldKnk (OLS-residualized)", "gf_resid", "st_resid"),
    ]:
        r_v, lo_v, hi_v = _boot_spearman(bias_ref[c1], bias_ref[c2], rng, B)
        rows.append({
            "category": "Inductive_Bias_Bootstrap", "comparison": label,
            "n_info": f"n={len(bias_ref)} shared genes", "metric": "Spearman rho",
            "estimate": r_v, "ci_95_low": lo_v, "ci_95_high": hi_v, "p_value": np.nan, "heterogeneity_I2_pct": np.nan,
        })

    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(os.path.join(OUT, "uncertainty_meta_and_ci.csv"), index=False)
    summary = {
        "seed": seed, "n_bootstrap": B,
        "meta_er_vs_normal_app4": {
            "wu_hedges_g": g_wu, "wu_var": v_wu, "pal_hedges_g": g_pal, "pal_var": v_pal, **meta_norm
        },
        "meta_er_vs_tnbc_app4": {
            "wu_hedges_g": g_wu_t, "wu_var": v_wu_t, "pal_hedges_g": g_pal_t, "pal_var": v_pal_t, **meta_tnbc
        },
        "rows": rows,
    }
    json.dump(summary, open(os.path.join(OUT, "uncertainty_meta_summary.json"), "w"), indent=2)
    print(f"Saved uncertainty_meta_and_ci.csv ({len(ci_df)} rows) and uncertainty_meta_summary.json")



def _require_path(path: str, dataset_hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(
            f"[DATA REQUIRED] Expected raw input not found: {path}\n"
            f"  Dataset: {dataset_hint}\n"
            f"  See README.md ('Pipeline Reproduction') for public accession download paths."
        )

if __name__ == "__main__":
    _require_path(os.path.join(ROOT, "data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq/metadata.csv"), "Wu et al. GSE176078 + LINCS L1000 GSE70138")
    finish_stage1_purity()
    run_inductive_bias_diagnostics()
    compute_uncertainty_and_meta()
