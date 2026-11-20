# NullPerturbBench

**Null-controlled cross-modal benchmarking of single-cell foundation models (Geneformer-V2-104M) and tensor-denoised GRN manifold perturbation (scTenifoldKnk) against empirical CRISPR, timepoint-matched chemical screens, and orthogonal MCF-7 RT-qPCR ground truth**

This repository provides the reproducible analysis, virtual perturbation, matched-null calibration, and figure-generation source code accompanying the manuscript:

> *Null-controlled benchmarking characterizes opposing inductive biases in zero-shot virtual perturbation models and validates pharmacological antigen-presentation activation in ER-positive breast cancer*

---

## Repository Structure

```text
NullPerturbBench/
├── code/
│   ├── 01_epithelial_app_multiomics.py       # Stage 1: Leukocyte-free & normal luminal scRNA-seq (BH-FDR) + TCGA/SCAN-B multi-omics
│   ├── 02_geneformer_epithelial_perturb.py   # Stage 2: Geneformer-V2-104M (median V2, 4096, <cls>/<eos>, L11) + 115 rank-matched nulls
│   ├── 03_sctenifold_epithelial_benchmark.py # Stage 3: scTenifoldKnk (10 pcNets + CP-ALS + directed Laplacian) + 100 nulls/KO
│   ├── 04_lincs_crispr_drug_groundtruth.py   # Stage 4: Landmark-aware LINCS L1000 MCF-7 CRISPR KOs, 1,760-drug & 24 h-matched evaluation
│   ├── 05_diagnostics_and_consistency.py     # Stage 5: Inductive-bias diagnostics, OLS residualization, Meta-analysis & Bootstrap 95% CIs
│   └── 06_make_figures_and_tables.py         # Stage 6: Renders Figures 1–6 (300-dpi PNG/PDF) & Supplementary Tables S1–S11
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

*(Curated derived source data tables covering Stages 1–5 are provided with the manuscript as Electronic Supplementary Material `Supplementary_Tables.xlsx`, Sheets S1–S11.)*

---

## Pipeline Reproduction

To run the end-to-end pipeline (Stages 1–6), clone the repository, install dependencies, place the required public datasets under `data/` (and Geneformer V2 weights under `~/.cache/geneformer/`), and execute `code/01` through `code/06` sequentially:

```bash
git clone https://github.com/zhangdaoyong/NullPerturbBench.git
cd NullPerturbBench
pip install -r requirements.txt

python code/01_epithelial_app_multiomics.py
python code/02_geneformer_epithelial_perturb.py
python code/03_sctenifold_epithelial_benchmark.py
python code/04_lincs_crispr_drug_groundtruth.py
python code/05_diagnostics_and_consistency.py
python code/06_make_figures_and_tables.py
```

### Public Data & Checkpoint Inputs

1. **Wu et al. (2021) Single-Cell Breast Cancer Atlas (GSE176078, GRCh38):** `data/processed/scrna_wu/Wu_etal_2021_BRCA_scRNASeq/` (`count_matrix_sparse.mtx`, `count_matrix_genes.tsv`, `count_matrix_barcodes.tsv`, `metadata.csv`).
2. **Pal et al. (2021) Human Breast Single-Cell Atlas (GSE161529, GRCh38):** Count matrices and `GSE161529_features.tsv.gz` filtered with dual epithelial/non-leukocyte molecular gating (`EPCAM/KRT18/KRT19 > 0` and `PTPRC/CD3D/CD68 == 0`) plus strict luminal sensitivity gating (`KRT18/19 > 0`, `KRT5/14/ACTA2 == 0`).
3. **TCGA-BRCA Multi-Omics (UCSC Xena, hg19 450k GPL16304) & SCAN-B (GSE96058 / GSE81538, GRCh38):** Harmonized RNA-seq, GISTIC2 copy-number calls, Illumina 450k methylation betas, and SCAN-B expression matrix under `data/processed/` and `data/raw/`.
4. **LINCS L1000 Phase II Level 5 MODZ (GSE70138):** `GSE70138_level5.gctx`, `gene_info.txt.gz`, and `sig_info.txt.gz` under `data/raw/mr/`.
5. **Geneformer Pre-Trained Checkpoint & Dictionaries:** Pre-trained `ctheodoris/Geneformer` (`Geneformer-V2-104M`, `token_dictionary_gc104M.pkl`, `gene_name_id_dict_gc104M.pkl`, and `gene_median_dictionary_gc104M.pkl`) under `~/.cache/geneformer/`.

---

## Citation & License

Released under the MIT License (`LICENSE`). Upon manuscript acceptance, a versioned release will be permanently archived on Zenodo with a persistent DOI. Data citations for all public cohorts (GSE176078, GSE161529, GSE70138, TCGA-BRCA, SCAN-B) and upstream frameworks (Geneformer, scTenifoldNet, scTenifoldKnk) are detailed in the accompanying manuscript.
