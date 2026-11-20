#!/usr/bin/env python3
"""Stage 4: Empirical LINCS L1000 (GSE70138) CRISPR & chemical
perturbation ground truth in MCF-7 and breast cell lines, with:
  - Explicit 24 h-matched compound enrichment alongside full 1,760-compound screen
  - 6 h vs 24 h kinetic breakdown across all 9 MCF-7 HDACi (including trichostatin-a)
  - Direct landmark (pr_is_lm=1) vs inferred (pr_is_lm=0) gene annotations
  - Formal Mann-Whitney tests for MCF-7 CRISPR HDAC1/2/3 vs other 50 CRISPR KOs
  - Paralog-expression percentile diagnostics evaluated against background KO distribution
  - Complete reporting of all 3 MCF-7 JAKi (ruxolitinib, baricitinib, tofacitinib)
  - Cross-modal alignment against Geneformer-V2-104M and scTenifoldKnk

Outputs under results/stage4_lincs/:
  - mcf7_crispr_53_app.csv
  - mcf7_compounds_1760_app.csv
  - mcf7_compounds_24h_app.csv
  - breast_celllines_hdaci_jaki_app.csv
  - cross_modal_benchmark_summary.json
"""
import json
import os
import h5py
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GCTX = os.path.join(ROOT, "data/raw/mr/GSE70138_level5.gctx")
GI = os.path.join(ROOT, "data/raw/mr/gene_info.txt.gz")
SIG = os.path.join(ROOT, "data/raw/mr/sig_info.txt.gz")
GF_DIR = os.path.join(ROOT, "results/stage2_geneformer")
ST_DIR = os.path.join(ROOT, "results/stage3_sctenifold")
OUT = os.path.join(ROOT, "results/stage4_lincs")
os.makedirs(OUT, exist_ok=True)

APP4 = ["HLA-A", "B2M", "TAP1", "PSMB8"]
APP8 = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9"]
KEY_COLS = APP8 + ["NLRC5", "CIITA", "STAT1", "IRF1", "ESR1", "HDAC1", "HDAC2", "HDAC3"]

DRUG_CLASSES = {
    "HDACi": ["romidepsin", "dacinostat", "jnj-26481585", "mocetinostat", "vorinostat",
              "belinostat", "entinostat", "tacedinaline", "trichostatin-a", "givinostat",
              "pracinostat", "apicidin", "scriptaid", "droxinostat", "tubastatin-a"],
    "JAKi": ["ruxolitinib", "tofacitinib", "baricitinib", "fedratinib", "momelotinib"],
    "DNMTi": ["decitabine", "azacitidine"],
    "SERM_SERD": ["raloxifene", "fulvestrant", "tamoxifen", "toremifene"],
}
DRUG_TO_CLASS = {d: cls for cls, ds in DRUG_CLASSES.items() for d in ds}


def main():
    gi = pd.read_csv(GI, sep="\t")
    sig = pd.read_csv(SIG, sep="\t", low_memory=False)
    id2sym = dict(zip(gi["pr_gene_id"].astype(int), gi["pr_gene_symbol"]))
    sym2id = dict(zip(gi["pr_gene_symbol"], gi["pr_gene_id"].astype(int)))
    lm_status = {
        g: {"is_landmark": int(gi.loc[gi["pr_gene_symbol"] == g, "pr_is_lm"].iloc[0]),
            "is_bing": int(gi.loc[gi["pr_gene_symbol"] == g, "pr_is_bing"].iloc[0])}
        for g in KEY_COLS if (gi["pr_gene_symbol"] == g).any()
    }

    with h5py.File(GCTX, "r") as f:
        row_ids = np.array([int(x) for x in f["/0/META/ROW/id"][:]])
        col_ids = [c.decode() if isinstance(c, bytes) else str(c) for c in f["/0/META/COL/id"][:]]
        col_map = {c: i for i, c in enumerate(col_ids)}
        row_map = {g: i for i, g in enumerate(row_ids)}
        all_row_syms = [id2sym.get(gid, str(gid)) for gid in row_ids]

        avail_keys = [g for g in KEY_COLS if g in sym2id and sym2id[g] in row_map]
        r_pairs = sorted([(row_map[sym2id[g]], g) for g in avail_keys])
        r_idx_sorted = [p[0] for p in r_pairs]
        r_syms_sorted = [p[1] for p in r_pairs]

        mat = f["/0/DATA/0/matrix"]
        key_mat = mat[:, r_idx_sorted]
        df_keys = pd.DataFrame(key_mat, index=col_ids, columns=r_syms_sorted)
        df_keys["APP4"] = df_keys[APP4].mean(axis=1)
        df_keys["APP8"] = df_keys[APP8].mean(axis=1)

        # 1. MCF-7 53 CRISPR Knockouts (trt_xpr in Cas9 sublines MCF7.101 / MCF7.311, 96 h)
        xpr = sig[sig["cell_id"].str.startswith("MCF7", na=False) & (sig["pert_type"] == "trt_xpr")].copy()
        xpr_k = df_keys.loc[xpr["sig_id"]].copy()
        xpr_k["pert_iname"] = xpr["pert_iname"].values
        xpr_grp = xpr_k.groupby("pert_iname")[avail_keys + ["APP4", "APP8"]].mean().reset_index()
        xpr_grp = xpr_grp.sort_values("APP4", ascending=False).reset_index(drop=True)
        xpr_grp["rank_APP4"] = np.arange(1, len(xpr_grp) + 1)
        xpr_grp["rank_PSMB8"] = xpr_grp["PSMB8"].rank(ascending=False).astype(int)
        xpr_grp.to_csv(os.path.join(OUT, "mcf7_crispr_53_app.csv"), index=False)

        # Full genome-wide profiles for HDAC1/2/3 CRISPR KOs
        crispr_full = {}
        for tg in ["HDAC1", "HDAC2", "HDAC3", "AKT1", "PIK3CA", "TP53", "MYC", "EGFR"]:
            sub_sids = xpr.loc[xpr["pert_iname"] == tg, "sig_id"].tolist()
            c_idx = sorted([col_map[s] for s in sub_sids if s in col_map])
            if c_idx:
                crispr_full[tg] = pd.Series(mat[c_idx, :].mean(axis=0), index=all_row_syms)

        # Genome-wide profiles for key drugs in MCF7
        mcf7_cp = sig[(sig["cell_id"] == "MCF7") & (sig["pert_type"] == "trt_cp")].copy()
        drug_full = {}
        for dg in ["romidepsin", "dacinostat", "JNJ-26481585", "vorinostat", "belinostat",
                   "mocetinostat", "entinostat", "ruxolitinib", "tofacitinib", "baricitinib",
                   "raloxifene", "decitabine"]:
            sub_sids = mcf7_cp.loc[mcf7_cp["pert_iname"].str.lower() == dg.lower(), "sig_id"].tolist()
            c_idx = sorted([col_map[s] for s in sub_sids if s in col_map])
            if c_idx:
                drug_full[dg] = pd.Series(mat[c_idx, :].mean(axis=0), index=all_row_syms)

    # Formal CRISPR statistics: HDAC1/2/3 (n=3) vs other 50 CRISPR KOs
    hdac_cr = xpr_grp[xpr_grp["pert_iname"].isin(["HDAC1", "HDAC2", "HDAC3"])]
    other_cr = xpr_grp[~xpr_grp["pert_iname"].isin(["HDAC1", "HDAC2", "HDAC3"])]
    crispr_stats = {}
    for col in ["APP4", "HLA-A", "B2M", "TAP1", "PSMB8", "STAT1", "HDAC1", "HDAC2", "HDAC3"]:
        _, p1 = stats.mannwhitneyu(hdac_cr[col], other_cr[col], alternative="greater")
        _, p2 = stats.mannwhitneyu(hdac_cr[col], other_cr[col], alternative="two-sided")
        crispr_stats[col] = {
            "hdac123_mean": float(hdac_cr[col].mean()),
            "other50_mean": float(other_cr[col].mean()),
            "other50_median": float(other_cr[col].median()),
            "other50_sd": float(other_cr[col].std()),
            "p_one_sided_greater": float(p1),
            "p_two_sided": float(p2),
        }
    # Individual paralog percentiles & relative z-scores vs other 50 CRISPR KOs
    paralog_diag = {}
    for src_g in ["HDAC1", "HDAC2", "HDAC3"]:
        row_s = xpr_grp[xpr_grp["pert_iname"] == src_g].iloc[0]
        paralog_diag[src_g] = {}
        for tgt_g in ["HDAC1", "HDAC2", "HDAC3"]:
            v = float(row_s[tgt_g])
            ref = other_cr[tgt_g]
            paralog_diag[src_g][tgt_g] = {
                "z": v,
                "pct_vs_other50": float((ref <= v).mean() * 100.0),
                "sd_vs_other50": float((v - ref.mean()) / ref.std()),
            }

    # 2A. MCF-7 All-Timepoint 1,760 Compounds summary
    cp_k = df_keys.loc[mcf7_cp["sig_id"]].copy()
    cp_k["pert_iname"] = mcf7_cp["pert_iname"].values
    cp_k["pert_itime"] = mcf7_cp["pert_itime"].values
    cp_grp = cp_k.groupby("pert_iname")[avail_keys + ["APP4", "APP8"]].agg(["mean", "count"])
    cp_flat = pd.DataFrame({col: cp_grp[(col, "mean")] for col in (avail_keys + ["APP4", "APP8"])})
    cp_flat["n_sigs"] = cp_grp[("APP4", "count")]
    cp_flat["drug_class"] = [DRUG_TO_CLASS.get(d.lower(), "Other") for d in cp_flat.index]
    cp_flat = cp_flat.sort_values("APP4", ascending=False).reset_index()
    cp_flat["rank_APP4"] = np.arange(1, len(cp_flat) + 1)
    cp_flat["pct_APP4"] = cp_flat["APP4"].rank(pct=True)
    cp_flat.to_csv(os.path.join(OUT, "mcf7_compounds_1760_app.csv"), index=False)

    hdaci_vals = cp_flat.loc[cp_flat["drug_class"] == "HDACi", "APP4"]
    other_vals = cp_flat.loc[cp_flat["drug_class"] == "Other", "APP4"]
    jaki_vals = cp_flat.loc[cp_flat["drug_class"] == "JAKi", "APP4"]
    _, p_hdac_1s = stats.mannwhitneyu(hdaci_vals, other_vals, alternative="greater")
    _, p_hdac_2s = stats.mannwhitneyu(hdaci_vals, other_vals, alternative="two-sided")

    # 2B. MCF-7 Timepoint-Matched 24 h Compounds summary (matches 24 h wet-lab benchmark)
    cp_24 = cp_k[cp_k["pert_itime"] == "24 h"]
    cp24_grp = cp_24.groupby("pert_iname")[avail_keys + ["APP4", "APP8"]].agg(["mean", "count"])
    cp24_flat = pd.DataFrame({col: cp24_grp[(col, "mean")] for col in (avail_keys + ["APP4", "APP8"])})
    cp24_flat["n_sigs_24h"] = cp24_grp[("APP4", "count")]
    cp24_flat["drug_class"] = [DRUG_TO_CLASS.get(d.lower(), "Other") for d in cp24_flat.index]
    cp24_flat = cp24_flat.sort_values("APP4", ascending=False).reset_index()
    cp24_flat["rank_APP4_24h"] = np.arange(1, len(cp24_flat) + 1)
    cp24_flat.to_csv(os.path.join(OUT, "mcf7_compounds_24h_app.csv"), index=False)

    h24_vals = cp24_flat.loc[cp24_flat["drug_class"] == "HDACi", "APP4"]
    o24_vals = cp24_flat.loc[cp24_flat["drug_class"] == "Other", "APP4"]
    _, p_hdac_24h_1s = stats.mannwhitneyu(h24_vals, o24_vals, alternative="greater")
    _, p_hdac_24h_2s = stats.mannwhitneyu(h24_vals, o24_vals, alternative="two-sided")

    # 6 h vs 24 h kinetics for all 9 HDACi and 3 JAKi in MCF-7
    kinetics = {}
    for d in sorted(cp_flat.loc[cp_flat["drug_class"].isin(["HDACi", "JAKi"]), "pert_iname"].tolist()):
        sub = cp_k[cp_k["pert_iname"] == d]
        kinetics[d] = {}
        for tm, grp in sub.groupby("pert_itime"):
            kinetics[d][tm] = {
                "n_sigs": int(len(grp)),
                "APP4_mean": float(grp["APP4"].mean()),
                "STAT1_mean": float(grp["STAT1"].mean()),
            }

    # 3. Multi-breast-cell-line HDACi vs JAKi vs DNMTi matrix
    breast_cells = ["MCF7", "HME1", "MCF10A", "BT20", "HS578T", "MDAMB231", "SKBR3"]
    cp_br = sig[sig["cell_id"].isin(breast_cells) & (sig["pert_type"] == "trt_cp")].copy()
    br_k = df_keys.loc[cp_br["sig_id"]].copy()
    br_k["pert_iname"] = cp_br["pert_iname"].values
    br_k["cell_id"] = cp_br["cell_id"].values
    br_k["drug_class"] = [DRUG_TO_CLASS.get(str(d).lower(), "Other") for d in br_k["pert_iname"]]
    br_sub = br_k[br_k["drug_class"] != "Other"].groupby(
        ["drug_class", "pert_iname", "cell_id"]
    )[APP4 + ["APP4", "APP8", "STAT1"]].mean().reset_index()
    br_sub.to_csv(os.path.join(OUT, "breast_celllines_hdaci_jaki_app.csv"), index=False)

    # 4. Cross-modal benchmarking: Geneformer & scTenifoldKnk vs L1000 Real CRISPR & HDACi
    alignment = {}
    for tg in ["HDAC1", "HDAC2", "HDAC3"]:
        if tg not in crispr_full:
            continue
        cr_s = crispr_full[tg].groupby(level=0).mean()
        gf_path = os.path.join(GF_DIR, f"gf_del_{tg}_gene_shifts.csv")
        st_path = os.path.join(ST_DIR, f"st_del_{tg}_gene_shifts.csv")
        rec = {}
        if os.path.exists(gf_path):
            gf = pd.read_csv(gf_path).set_index("gene")["mean_shift"]
            shared = gf.index.intersection(cr_s.index)
            r_abs, p_abs = stats.spearmanr(gf.loc[shared], cr_s.loc[shared].abs())
            r_sign, p_sign = stats.spearmanr(gf.loc[shared], cr_s.loc[shared])
            top100_cr = set(cr_s.loc[shared].abs().nlargest(100).index)
            top100_gf = set(gf.loc[shared].nlargest(100).index)
            rec["gf_vs_crispr_abs_rho"] = float(r_abs)
            rec["gf_vs_crispr_abs_p"] = float(p_abs)
            rec["gf_vs_crispr_signed_rho"] = float(r_sign)
            rec["gf_vs_crispr_top100_recall"] = len(top100_cr & top100_gf) / 100.0
        if os.path.exists(st_path):
            st = pd.read_csv(st_path).set_index("gene")["displacement"]
            shared = st.index.intersection(cr_s.index)
            r_abs, p_abs = stats.spearmanr(st.loc[shared], cr_s.loc[shared].abs())
            r_sign, p_sign = stats.spearmanr(st.loc[shared], cr_s.loc[shared])
            top100_cr = set(cr_s.loc[shared].abs().nlargest(100).index)
            top100_st = set(st.loc[shared].nlargest(100).index)
            rec["st_vs_crispr_abs_rho"] = float(r_abs)
            rec["st_vs_crispr_abs_p"] = float(p_abs)
            rec["st_vs_crispr_signed_rho"] = float(r_sign)
            rec["st_vs_crispr_top100_recall"] = len(top100_cr & top100_st) / 100.0
        alignment[f"CRISPR_{tg}"] = rec

    combo_gf_path = os.path.join(GF_DIR, "gf_del_HDAC1_HDAC2_HDAC3_gene_shifts.csv")
    combo_st_path = os.path.join(ST_DIR, "st_del_HDAC1_HDAC2_HDAC3_gene_shifts.csv")
    for dg in ["romidepsin", "dacinostat", "JNJ-26481585", "ruxolitinib"]:
        if dg not in drug_full:
            continue
        dr_s = drug_full[dg].groupby(level=0).mean()
        rec_d = {}
        for tg in ["HDAC1", "HDAC2", "HDAC3"]:
            cr_s = crispr_full[tg].groupby(level=0).mean()
            sh = dr_s.index.intersection(cr_s.index)
            r_d, p_d = stats.spearmanr(dr_s.loc[sh], cr_s.loc[sh])
            rec_d[f"vs_CRISPR_{tg}_rho"] = float(r_d)
            rec_d[f"vs_CRISPR_{tg}_p"] = float(p_d)
        if os.path.exists(combo_gf_path):
            gf_c = pd.read_csv(combo_gf_path).set_index("gene")["mean_shift"]
            sh = dr_s.index.intersection(gf_c.index)
            r_c, p_c = stats.spearmanr(gf_c.loc[sh], dr_s.loc[sh].abs())
            rec_d["vs_GF_combo3_abs_rho"] = float(r_c)
            rec_d["vs_GF_combo3_abs_p"] = float(p_c)
        if os.path.exists(combo_st_path):
            st_c = pd.read_csv(combo_st_path).set_index("gene")["displacement"]
            sh = dr_s.index.intersection(st_c.index)
            r_c, p_c = stats.spearmanr(st_c.loc[sh], dr_s.loc[sh].abs())
            rec_d["vs_ST_combo3_abs_rho"] = float(r_c)
            rec_d["vs_ST_combo3_abs_p"] = float(p_c)
        alignment[f"Drug_{dg}"] = rec_d

    summary = {
        "n_mcf7_crispr_genes": int(len(xpr_grp)),
        "crispr_cell_sublines": sorted(xpr["cell_id"].unique().tolist()),
        "crispr_timepoint": sorted(xpr["pert_itime"].unique().tolist()),
        "n_mcf7_compounds": int(len(cp_flat)),
        "hdaci_n_in_mcf7": int((cp_flat["drug_class"] == "HDACi").sum()),
        "hdaci_mean_app4": float(hdaci_vals.mean()),
        "hdaci_median_app4": float(hdaci_vals.median()),
        "other_mean_app4": float(other_vals.mean()),
        "jaki_mean_app4": float(jaki_vals.mean()),
        "jaki_all_mcf7": cp_flat[cp_flat["drug_class"] == "JAKi"][["pert_iname", "rank_APP4", "APP4", "STAT1"]].to_dict(orient="records"),
        "p_hdaci_vs_other_app4": float(p_hdac_1s),
        "p_hdaci_vs_other_app4_twosided": float(p_hdac_2s),
        "mcf7_24h_matched": {
            "n_compounds_24h": int(len(cp24_flat)),
            "hdaci_n_24h": int((cp24_flat["drug_class"] == "HDACi").sum()),
            "hdaci_mean_app4_24h": float(h24_vals.mean()),
            "other_mean_app4_24h": float(o24_vals.mean()),
            "p_one_sided_24h": float(p_hdac_24h_1s),
            "p_two_sided_24h": float(p_hdac_24h_2s),
        },
        "hdaci_jaki_timepoint_kinetics": kinetics,
        "landmark_status": lm_status,
        "mcf7_crispr_hdac_vs_other50": crispr_stats,
        "mcf7_crispr_paralog_diagnostics": paralog_diag,
        "mcf7_crispr_hdac_median_app4": float(hdac_cr["APP4"].mean()),
        "mcf7_crispr_all_median_app4": float(xpr_grp["APP4"].median()),
        "mcf7_compounds_psmb8_median": float(cp_flat["PSMB8"].median()),
        "cross_modal_alignment": alignment,
    }
    json.dump(summary, open(os.path.join(OUT, "cross_modal_benchmark_summary.json"), "w"), indent=2)
    print(f"LINCS Stage 4 Summary:")
    print(f"  All timepoints (n={len(cp_flat)}): HDACi (n={summary['hdaci_n_in_mcf7']}) mean={summary['hdaci_mean_app4']:.3f} vs Other={summary['other_mean_app4']:.3f} (1s P={p_hdac_1s:.4g}, 2s P={p_hdac_2s:.4g})")
    print(f"  24 h-matched   (n={len(cp24_flat)}): HDACi (n={len(h24_vals)}) mean={h24_vals.mean():.3f} vs Other={o24_vals.mean():.3f} (1s P={p_hdac_24h_1s:.4g}, 2s P={p_hdac_24h_2s:.4g})")
    print(f"  CRISPR HDAC1/2/3 vs other 50: APP4 1s P={crispr_stats['APP4']['p_one_sided_greater']:.4g}, PSMB8 1s P={crispr_stats['PSMB8']['p_one_sided_greater']:.4g} (bg med={crispr_stats['PSMB8']['other50_median']:.3f})")



def _require_path(path: str, dataset_hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(
            f"[DATA REQUIRED] Expected raw input not found: {path}\n"
            f"  Dataset: {dataset_hint}\n"
            f"  See README.md ('Pipeline Reproduction') for public accession download paths."
        )

if __name__ == "__main__":
    _require_path(os.path.join(ROOT, "data/raw/mr/GSE70138_level5.gctx"), "LINCS L1000 Phase II GSE70138")
    main()
