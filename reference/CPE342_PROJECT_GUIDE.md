# CPE342 Machine Learning Project — Codex Development Guide

> Source: `CPE342_Project_Instruction(1).pdf` (28 pages), CPE342 Machine Learning (1/2026), KMUTT.
> This document summarizes the assignment and gives an actionable implementation plan for a 5-member team.
> **Important:** Sections marked **[Required]** reflect the assignment. Sections marked **[Recommended]** are implementation suggestions, not instructor mandates.

## 1. Project Overview [Required]

**Project:** Karena Player Intelligence System  
**Scenario:** The team acts as ML Engineers at Karena Thailand, a fictional game publishing platform.  
**Mission:** Develop five ML tasks as part of a Player Intelligence / Lifetime Value (LTV) Prediction System, supporting game integrity, player understanding, monetization, and retention.  
**Team size:** 5 members.  
**Deliverables:** Kaggle predictions, Technical Report, Source Code.

> Scope clarification: The assignment specifies **five prediction tasks combined into one Kaggle submission**. It does **not explicitly require** a sixth LTV model, frontend, backend API, dashboard, or production deployment.

## 2. Grading and Deadlines [Required]

| Component | Weight |
|---|---:|
| Kaggle leaderboard performance | 70% |
| Technical Report | 20% |
| Source Code | 10% |

| Date (2026) | Event |
|---|---|
| November 2 | Competition release |
| November 24 | Kaggle competition closes |
| November 30 | Technical Report and Source Code due |

The method/location for Report and Source Code submission will be announced on **LEB2**.

### Kaggle baselines (public weighted score)

- Ground baseline: **0.68**.
- Challenge baseline: **0.75**.
- Each task is evaluated separately, then scores are combined using a weighted average: `sum(w_i * S_i) / sum(w_i)`.
- **Task weights are not stated in the PDF.** Check the Kaggle competition page for actual settings.
- Page 7 describes grade conversion relative to baseline, including an **example** with `alpha = 60%`; do not assume the example alpha is the final configured value.

## 3. Required ML Tasks [Required]

| Task | Title | Learning type | Target | Evaluation metric |
|---|---|---|---|---|
| 1 | Anti-Cheat Pre-Filter | Binary classification | `is_cheater`: 0 legitimate, 1 cheater | **F2** |
| 2 | Player Segment Classification | Multi-class classification | `segment`: 0 Casual, 1 Grinder, 2 Social, 3 Whale | **F1** |
| 3 | Player Monthly Spending Prediction | Regression | `spending_30d` in THB | **Normalized MAE score** |
| 4 | Game Title Detection | Image classification (5 classes) | `game_title` | **Macro F1** |
| 5 | Account Security Monitoring | Unsupervised anomaly detection | `is_anomaly`: 0 normal, 1 anomalous | **F3** |

### Task 1: Anti-Cheat Pre-Filter

- **Goal:** Flag high-probability cheaters for priority manual review.
- **Data:** 33 total features (32 predictors + target), as described in the slides.
- **Example features:** `kill_death_ratio`, `headshot_percentage`, `win_rate`, `accuracy_score`, `reaction_time_ms`.
- **Metric:** `F2 = 5PR / (4P + R)`; recall receives more weight than precision.
- **Hints from slides:** handle missing values and class imbalance, consider resampling/class weights/SMOTE, feature interactions, cross-validation, threshold tuning, ensemble methods.

### Task 2: Player Segment Classification

- **Goal:** Predict player behavioral segment for personalized marketing.
- **Data:** 45 total features (44 predictors + target).
- **Example features:** `play_frequency`, `avg_session_duration`, `total_spending_thb`, `avg_monthly_spending`, `friend_count`.
- **Metric:** F1 (the PDF calls it F1; it does **not specify** the averaging variant for Task 2).
- **Hints from slides:** address missing data / imbalance, stratified K-fold, test XGBoost, LightGBM or CatBoost, feature ratios/interactions, stacking.

### Task 3: Player Monthly Spending Prediction

- **Goal:** Predict a player's in-game spending in the next 30 days in THB.
- **Data:** 33 total features (32 predictors + target).
- **Example features:** `friend_count`, `event_participation_rate`, `daily_login_streak`, `avg_session_length`, `historical_spending`.
- **Metric:** `NMAE_score = 1 / (1 + sum(abs(y - y_pred)) / sum(y))` (as shown in the PDF).
- **Hints from slides:** account for many zero-spending players; test a two-stage approach (spend or not, then amount), target log transformation, gradient-boosting regressors, ensembling. Handle inverse target transformations correctly.

### Task 4: Game Title Detection

- **Goal:** Classify the game depicted in a screenshot.
- **Target classes:** `7-11K`, `fiveN`, `freefried`, `pube`, `roblock`.
- **Images:** The PDF describes gameplay screenshots at 144p, center-cropped to 230 × 120; inspect actual downloaded image sizes and structure.
- **Metric:** **Macro F1**.
- **Hints from slides:** image augmentation, class imbalance, CNN cross-validation, ensembling, VGG16/VGG19, ViT, Swin Transformer.
- **Special exception:** Pre-trained models **are allowed** for Task 4.
- **Compute:** GPU may be useful (Kaggle / Colab). GPU quota figures in slides are illustrative and may change.

### Task 5: Account Security Monitoring

- **Goal:** Flag unusual account behavior without using labeled training data.
- **Target in submission:** `is_anomaly`: 0 normal, 1 anomalous.
- **Example features:** time-windowed `login_count`, `login_lat/lon`, `location_changes`, `device_count` (suffixed `_1` ... `_4`).
- **Metric:** `F3 = 10PR / (9P + R)`.
- **Data note from slides:** **No labeled training set is provided; `test.csv` only.**
- **Hints from slides:** Isolation Forest, One-Class SVM, autoencoder; unsupervised ensembles, soft/hard voting, contamination and decision-threshold tuning.
- **Rule:** Do not use hidden test labels or pseudo-label/fine-tune against test predictions. Carefully reconcile the Task 5 input arrangement with the official Kaggle competition instructions while respecting the ban on test-set training.

## 4. Competition Rules [Required]

1. Use **only the provided datasets**.
2. The slides allow combining **Train + Validation** for final training.
3. **Do not use the test set for training**, pseudo-labeling, or fine-tuning.
4. **Only Task 4 may use pre-trained models**; Tasks 1, 2, 3, and 5 must train from scratch.
5. **AutoML frameworks are prohibited**; hyperparameter-tuning tools are allowed.
6. Feature engineering and ensembling are encouraged.
7. Work with your own team only. Discussions of concepts are fine; **do not share code or predictions with other teams**.
8. All five team members must make meaningful contributions.

> Do not interpret "train from scratch" as banning conventional libraries such as scikit-learn / XGBoost / LightGBM / CatBoost. It means models must be fitted from the permitted provided data, without importing a pre-trained model for Tasks 1, 2, 3, or 5.

## 5. Competition Enrollment [Required]

- [ ] Fill team information in the designated Google Sheet.
- [ ] Create/log into individual Kaggle accounts.
- [ ] Join the competition using the instructor-provided Kaggle link.
- [ ] Set the Kaggle team name to the **exact Team ID** from the Google Sheet (example: `ML001`).
- [ ] Invite all team members to the Kaggle team.
- [ ] Download the provided datasets and `sample_submission.csv`.

## 6. Kaggle Submission Format [Required]

Fill the official `sample_submission.csv` with predictions from **all five tasks**.

- **Columns:** `id,task1,task2,task3,task4,task5`.
- **Shape shown in the slides:** **25,889 rows × 6 columns** (includes header only as CSV metadata; the number is data rows).
- Every prediction column must be filled for **every** row.
- Preserve the exact IDs, row order, column names, and sample-file format.
- Output types:
  - `task1`: integer 0 or 1
  - `task2`: integer 0, 1, 2, or 3
  - `task3`: numeric spending prediction (THB)
  - `task4`: one of `7-11K`, `fiveN`, `freefried`, `pube`, `roblock`
  - `task5`: integer 0 or 1
- Validate there are no missing values, invalid classes, row-count mismatches, or misaligned IDs.

Illustrative rows from the instruction slides (not actual predictions):

```csv
id,task1,task2,task3,task4,task5
ANS00001,0,2,0,freefried,1
ANS00002,1,3,10000,fiveN,0
```

## 7. Technical Report [Required content; recommended organization]

The slides require a **Technical Report**, in Thai or English, covering:

- Methodology for **each task**: EDA, preprocessing, model design, results.
- Insights and common mistakes.
- Domain interpretation: business implications.
- Team collaboration: roles and takeaways.
- Conclusion: lessons learned, ML insights, future improvements.

**Suggested outline:**

1. Introduction / Business Problem
2. Dataset Overview and Data Quality
3. Task 1 — Methodology, Experiments, Results
4. Task 2 — Methodology, Experiments, Results
5. Task 3 — Methodology, Experiments, Results
6. Task 4 — Methodology, Experiments, Results
7. Task 5 — Methodology, Experiments, Results
8. Cross-task Insights / Business Interpretation
9. Team Contributions and Collaboration
10. Conclusion, Limitations, Future Improvements

For each task, record data splits, preprocessing, feature engineering, model choices, validation strategy, metrics, tuning, errors/limitations, and reproducibility details.

## 8. Source Code and Repository [Required + Recommended]

**Required:** Create a GitHub repository and organize code in task-specific folders (`Task1`, `Task2`, etc.).

**Recommended structure (adapt to real downloaded data):**

```text
CPE342-ML-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── Task1/
│   ├── train.ipynb
│   └── predict.py
├── Task2/
│   ├── train.ipynb
│   └── predict.py
├── Task3/
│   ├── train.ipynb
│   └── predict.py
├── Task4/
│   ├── train.ipynb
│   └── predict.py
├── Task5/
│   ├── experiment.ipynb
│   └── predict.py
├── scripts/
│   ├── merge_submission.py
│   └── validate_submission.py
├── submissions/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

Recommended engineering practices:

- Use one predictable interface per Task to generate a prediction vector aligned to `sample_submission.csv`.
- Fit preprocessing on allowed training partitions only, to avoid leakage.
- For Tasks 1–4, validate before training the final allowed model on Train + Validation.
- Record model versions, random seeds, metric calculations, and run commands.
- Do not commit large datasets, secrets, or model weights unless necessary and permitted.
- Keep notebooks readable and ensure prediction scripts can reproduce output.
- For Task 5, confirm how to perform an unsupervised fit without violating the separate test-set training restriction; do not invent labeled validation data.

## 9. Suggested Team Allocation [Recommended]

| Member | Primary responsibility |
|---|---|
| 1 | Task 1 — Anti-Cheat |
| 2 | Task 2 — Segmentation |
| 3 | Task 3 — Spending Regression |
| 4 | Task 4 — Image Classification |
| 5 | Task 5 — Anomaly Detection |

Shared responsibilities: Kaggle enrollment, initial submission, CSV integration, GitHub hygiene, code review, report writing, and final verification. Assign **one submission integrator** and **one report coordinator**; those roles may overlap with Task ownership.

Task difficulty estimates are suggestions, not stated by the instructor. Task 4 may need more GPU/experimentation; Task 5 has an unusual unlabeled-data setup.

## 10. Development Workflow [Recommended]

Repeat for each task where applicable:

1. Inspect dataset files, schema, sample counts, feature types, target distribution (if labels are available), missingness, duplicates, and leakage risks.
2. Establish a simple valid baseline.
3. Build a train/validation evaluation procedure using the task-specific metric.
4. Apply task-appropriate preprocessing and feature engineering.
5. Train candidate models, compare validation metrics, and tune conservatively.
6. Save best hyperparameters and reproducible training instructions.
7. Generate prediction outputs for required submission IDs.
8. Integrate all five outputs, validate combined CSV, and submit to Kaggle.
9. Document experiments, scores, error analysis, and domain/business interpretation.

**Task-specific emphasis:**

- T1: imbalance, recall, threshold optimization for F2.
- T2: multiclass stratification and confirm F1 averaging with competition scorer.
- T3: skewed/zero-inflated spending, nonnegative outputs, inverse transforms.
- T4: permitted pretraining, image transforms, macro F1, consistent class labels.
- T5: genuine unsupervised methodology; contamination/threshold selection without test labels.

## 11. Suggested Timeline [Recommended]

| Date | Suggested activity |
|---|---|
| Nov 15–16 | Confirm team, download data, repo setup, EDA |
| Nov 17–18 | Obtain working baseline predictions for all 5 tasks |
| Nov 19 | First complete Kaggle submission |
| Nov 20–22 | Improve features, validation, hyperparameters, ensembles |
| Nov 23 | Freeze / verify the best reproducible submission |
| Nov 24 | Kaggle competition closes |
| Nov 25–29 | Finish report, review code, verify reproducibility |
| Nov 30 | Submit report + source code |

## 12. Definition of Done / Master Checklist

### Enrollment
- [ ] Team registered with exact Team ID.
- [ ] All five members added to Kaggle team.
- [ ] Competition datasets and sample submission obtained.

### Modeling
- [ ] Task 1: valid F2-evaluated binary model and predictions.
- [ ] Task 2: valid F1-evaluated 4-class model and predictions.
- [ ] Task 3: valid normalized-MAE-evaluated spending model and predictions.
- [ ] Task 4: valid macro-F1-evaluated image model and predictions.
- [ ] Task 5: valid unsupervised anomaly approach and predictions, compliant with test-data rules.
- [ ] No prohibited datasets, test-set training, or AutoML.

### Submission
- [ ] Output matches official `sample_submission.csv`.
- [ ] 25,889 IDs/rows (verify against downloaded official file).
- [ ] Exactly six columns in correct order.
- [ ] All five task columns have valid labels/values and no missing predictions.
- [ ] Kaggle accepted submission before November 24.
- [ ] Public leaderboard score recorded.

### Report and Code
- [ ] Report includes each task's methodology and results.
- [ ] Report includes insights, business meaning, team contribution, and conclusions.
- [ ] GitHub repository has organized Task1–Task5 source code.
- [ ] README describes setup and reproduction.
- [ ] Final report and source code submitted by November 30 via announced method.

## 13. Instructions for Codex Agent [Recommended]

When implementing this project:

1. **Read this guide**, the official PDF, the dataset folders, and the competition page before modifying code.
2. Treat PDF/competition rules as the source of truth. Do **not** assume unmentioned dataset names, exact feature schemas, submission paths, metric averaging settings, or task weights.
3. Never train on competition test data or use pseudo-labeling/fine-tuning on it. Respect the Task 5 unlabeled-data exception without assuming it overrides the global rule; surface any conflict for human review.
4. Do not use pre-trained models outside Task 4 or AutoML frameworks anywhere.
5. Prefer small, testable scripts and reproducible experiments over complex architecture.
6. Start by checking repository state, dataset availability, and expected output format; then produce a prioritized implementation plan.
7. Do not invent evaluation scores or claim that a Kaggle submission succeeded without real evidence.
8. Avoid building a UI, API, dashboard, or sixth LTV model unless explicitly requested.

### Starter prompt for Codex

```text
Read CPE342_PROJECT_GUIDE.md and inspect the current repository structure and datasets.
First, summarize which of Tasks 1–5 are implemented and what is missing.
Then propose the smallest rule-compliant plan to get a valid baseline submission for all five tasks.
Do not use test data for training, do not use AutoML, and only permit pre-trained models in Task 4.
Before editing, list the files you plan to create or modify and any conflicts or unknowns in the assignment.
Prioritize correct dataset alignment, reproducible evaluation, and a validated combined Kaggle CSV.
```

---

**Primary reference:** `CPE342_Project_Instruction(1).pdf` — notably pp. 3–7 (team/submission/grading), p. 12 (five-task mission), p. 13 (rules), pp. 14–23 (tasks), p. 24 (submission format), pp. 25–27 (metrics, score, baselines).
