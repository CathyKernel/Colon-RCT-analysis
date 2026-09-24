# Data Codebook and Provenance

**Project:** Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
**Author:** Cathy

## 1. Provenance

The analysis dataset derives from `data/raw/colon.csv`, the `colon` dataset
distributed with the R `survival` package (Therneau 2024), mirrored by the
Rdatasets project (https://vincentarelbundock.github.io/Rdatasets/). The
underlying trial was conducted by the North Central Cancer Treatment Group
and reported by Laurie et al. (1989) and Moertel et al. (1990). The dataset
contains 1,858 rows (929 patients x 2 event types) and is redistributed here
for research and educational purposes with attribution.

The raw file stores **two rows per patient**: `etype = 1` records time to
tumour recurrence (death without recurrence treated as censoring), and
`etype = 2` records time to death. Covariates are duplicated across the two
rows; `R/01_data_curation.R` verifies their consistency and pivots the data
to one row per patient.

## 2. Analysis file: `data/processed/colon_analysis.csv`

One row per patient (n = 929). Times are in years unless stated.

| Variable | Type | Description |
|---|---|---|
| `id` | integer | patient identifier |
| `rx` | factor | randomized arm: `Obs`, `Lev`, `Lev+5FU` |
| `sex` | factor | `Male`, `Female` |
| `age` | numeric | age in years |
| `age_group` | factor | `< 60`, `60-69`, `>= 70` (derived) |
| `obstruct` | factor | obstruction of colon by tumour: `No`, `Yes` |
| `perfor` | factor | perforation of colon: `No`, `Yes` |
| `adhere` | factor | adherence to nearby organs: `No`, `Yes` |
| `nodes` | numeric | number of positive lymph nodes (18 missing) |
| `nodes_cat` | factor | `0`, `1-3`, `>= 4` (derived) |
| `differ` | factor | tumour differentiation: `Well`, `Moderate`, `Poor` (23 missing) |
| `extent` | factor | local spread: `Submucosa`, `Muscle`, `Serosa`, `Contiguous structures` |
| `surg` | factor | surgery-to-registration interval category: `Short`, `Long` |
| `node4` | factor | indicator of >= 4 positive nodes: `No`, `Yes` |
| `dfs_time` | numeric | **primary endpoint** time: years to recurrence or death, whichever occurs first |
| `dfs_status` | integer | **primary endpoint** event: 1 = recurrence or death (506 events) |
| `rec_time` | numeric | years to recurrence (secondary coding) |
| `rec_status` | integer | 1 = recurrence (468 events) |
| `rec_cr` | integer | competing-risk coding: 0 = censored (423), 1 = recurrence (468), 2 = death without recurrence (38) |
| `os_time` | numeric | **secondary endpoint** time: years to death from any cause |
| `os_status` | integer | **secondary endpoint** event: 1 = death (452 events) |

## 3. Derived files

| File | Contents |
|---|---|
| `data/processed/colon_mi_dfs.rds` | `mice` multiple-imputation object (m = 20) for the composite DFS endpoint, constructed with the Nelson-Aalen cumulative hazard and event indicator in the imputation models (White & Royston 2009) |
| `data/processed/colon_mi_os.rds` | the same for the overall-survival endpoint |

## 4. Endpoint construction note

The published `etype = 1` records treat death without recurrence as
censoring. This project additionally constructs the composite DFS endpoint
(`dfs_time`, `dfs_status`) because modern adjuvant colorectal trials define
DFS as recurrence or death. Both views are retained so that:

- the primary analysis uses the composite DFS endpoint (506 events);
- a competing-risk sensitivity analysis (Aalen-Johansen cumulative incidence
  and Fine-Gray subdistribution hazards) handles the 38 deaths without
  recurrence explicitly on the recurrence endpoint.

## 5. Original variable coding (raw file)

`sex` 1 = male; `obstruct`/`perfor`/`adhere` 1 = yes; `differ` 1 = well,
2 = moderate, 3 = poor; `extent` 1 = submucosa, 2 = muscle, 3 = serosa,
4 = contiguous structures; `surg` 1 = short, 0 = long surgery-to-registration
interval; `status` 1 = event; `time` in days; `etype` 1 = recurrence,
2 = death.
