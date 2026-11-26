# Task2 TabM scratch pilot + fixed LR blend — 14 November 2026

User explicitly authorized this arm after Task1 weights, Task3 conditional stack,
and Task2 XGB/LR failed their unchanged practical guards. Read current handoff,
competition rules/all Task2 hints, Task2 README, roadmap and latest XGB report.
Best actual public aggregate .72430; baseline .72601/gap .00171. This is NOT Task2 F1.

## Problem, novelty, complete teacher-hint status

XGB+LR gains .001864526 on all5 folds, below .002; standalone XGB .748701.
Prior nonlinear LR/LGB/Cat/stack/scaler/QDA completed. New hypothesis: one
scratch-trained neural representation can add complementary errors to raw LR.
No claim that TabM beats trees/LR on our data or that past small gains prove optimum.

| Hint | Status/action |
|---|---|
| Missing values | Existing fit-only LR median+indicators; reuse its exact fold-local transform for TabM, no new imputation sweep |
| Imbalance | Earlier Cat weighting/decision multipliers done. Natural prevalence, unweighted CE; fixed old multipliers, no new tuning. SMOTE/resampling deferred to isolate representation and avoid mixed-type semantics |
| Stratified K-Fold | Five historical outer folds preserved; all new epoch selection inside outer-fit; no outer threshold/weight search |
| XGB/LGB/Cat | All latest bounded arms completed; this new neural family is additional to hints, not a substitute for considering them |
| Stacking/ensemble | Earlier nested stack done; now ONE fixed primary .75LR+.25TabM, no meta-model/alpha search |
| Interactions/ratios | Earlier quadratic/targeted work done; retain44 raw features, no new handcrafted interactions |

## Fixed architecture/training and resources

Official TabM0.0.3 +rtdl_num_embeddings0.0.12 pure-Python wheels, exact SHA in
manifest, licenses preserved. No external/pretrained weights or data. Do not
install over local environment; wheels imported privately/offline. Torch already
provided by runtime; record actual versions/devices, never claim cloud run locally.

One architecture: TabM.make(all encoded inputs as numeric, d_out4, arch_type='tabm',
n_blocks2,d_block256,dropout.1,k16,start_scaling_init='random-signs',num_embeddings=None).
All args explicit. No embedding/width/depth/seed search. Mean cross-entropy across
ALL per-member logits (N,K,4), NOT CE of mean logits. Inference mean of16 softmax
probabilities, NOT softmax of mean logits. AdamW lr.002/weight_decay.0003,
batch512,no dropped training rows,FP32,no AMP/TF32/torch.compile. Max80epochs,
patience12, choose primary-blend stop MacroF1, ties higher TabM log-loss preference
NOT used: earliest best F1 retained. Refit from fresh initialization for chosen
epoch count with all fit rows, no stopping on outer/audit/test.

Full run requires actual CUDA >=4GiB and one successful forward/backward
preflight. Use cuda0 only, even if2T4 visible; no pooled-VRAM or quota promise.
CPU-only machine refuses full training before outputs/model fits. CPU smoke is
runtime evidence only: same architecture, small proper-role train-row samples,
2epochs/batch32. No quality/audit/test/candidate from smoke. Record peak GPU memory.

## Inputs, preprocessing and LR controls prepared locally

Current Task2 train104,000 SHA0cdaa5fad6c17a7f27d965b988cd72f6113a1a53c1f88bcf5d12c25128249f48.
Saved88,400 dev/15,600 reused audit; original5 outer fold IDs/order/targets fixed.
Test raw bytes included for transport only; no test CSV parsing/feature transform/
inference until BOTH development/audit guards pass. Integrity hashes necessarily
read test bytes while packaging/transport; this is not test-derived model fitting.

Original44-feature/parser semantics retained, including literal None→missing.
Freeze pandas original NA token policy. Portable standard-library CSV reader and
NumPy transform implement existing LR median+missing indicators+StandardScaler+
categorical OHE exactly. Unknown categories all-zero, raw numeric infinities→NaN;
preferred_game_mode finite numeric→integer string; missing→__MISSING__.
Original original numeric/categorical column order and fit-only medians/indicator
indices/scaler moments/OHE categories stored per role. TabM receives that same
dense transform cast float32. LR float64 softmax coefficients stay untouched.

Role-specific baseline LR assets are controls, never TabM warm weights. Reuse
verified prior pilot-A/outer-T/development/final LR artifacts where exact fit rows
match; locally fit6 new inner-stop LR controls (5 outer-inner +conditional dev-inner),
all from proper training rows, raw C.5/lbfgs/max1600/converged. These possible-stage
training-only control assets are prepared BEFORE cloud pilot, explicitly; no new
TabM audit/final training or audit performance evaluation during preparation.
Independently reconstruct every prep statistic/categories on its declared fit rows.
Portable LR probabilities must match sklearn controls <=1e-8 and historical OOF/
audit controls <=1e-8 (labels exact). Existing final historical model may be read,
but new final/test inference waits for guards. No sklearn/pandas/joblib needed on
cloud; avoids silently retraining a changed LR under another package version.

## Pilot compute screen and five-fold primary

Pilot is entirely inside historical OUTER0 FIT T0 (outer0 query excluded).
Same saved new XGB-style stratified A64%/B16%/C20% split seed20261114/15.
Pilot TabM trains A, selects epoch using B primary-blend F1 and A-only LR transform/
probabilities. Evaluate C ONCE with selected epoch. C is a compute-screen query,
not training/stopping and not a new independent dataset.
Predeclared screen: fixed primary C F1 >=LR C F1, plus unattainable correctness
oracle improvement>=.002, and TabM corrects at least max(20,ceil(.01*LR-errors))
LR errors on C. Oracle uses labels diagnostically only and is not deployable,
not replacement quality evidence. If screen fails: stop this arm, no5fold/audit/
test/candidate or automatic new config. Pilot pass is NOT passing the .002 guard.

If pilot passes: for each outer-fit T, StratifiedShuffleSplit seed20261114,
80%inner-fit/20%inner-stop (neither sees outer query). Fit one TabM/inner-fit-local
prep; select epochs by fixed .75LR+.25TabM old-multiplier MacroF1 on inner-stop.
Fresh TabM refit on all T, using T-local prep and selected epoch count; evaluate
outer query ONCE. No architecture/alpha/multiplier selection on outer. Seeds are
role-fixed in manifest. LR control probabilities frozen and checked. Main dev
TabM budget pilot1 +5×(inner+refit)=11 fits maximum. All full training rows used.

Require primary OOF MacroF1 gain>=.002 vs old LR AND mean paired-fold gain>one SE.
Same original practical filter; no lowering after results, no statistical-test
claim. Only if passes: repeat inner80/20 epoch selection on dev, refit TabM on all
dev, check reused audit regression<=.001. Only if passes: one final TabM scratch
fit on all104,000 for frozen dev epochs, then first test table transform/inference
and one Task2-only CSV. Max14TabM fits; no new fit outside this fixed recipe.

Historical folds/base choices/weights/previously inspected rows have selection
history; neither pilot nor new folds create independent unseen data. Reused audit
is a regression guard, not independent holdout. OOF→refit/sample-size shift remains.

## Integrity/replay/export and Kaggle delivery

Protect40scoredCSVs, originalTask1/2/3 models/records and existing packets.
Immutable packet: raw CSVs, splits/roles, portable LR states/golden probabilities,
source/protocol/config, vendor wheels, current best CSV/SHA, inventory manifest.
Notebook pins manifest SHA; supports intact ZIP OR Kaggle-expanded same-byte files.
Offline wheel imports; no installs on Kaggle/network dependency. No credentials.
User attaches new PRIVATE Dataset ZIP and chooses GPU/RunAll; no agent upload.

Every epoch persist history/last model+optimizer+CPU/CUDA RNG and identity hashes;
best tensor-only checkpoint, samples with exact raw row identities and encoded
inputs/outputs retained. Resume only unchanged packet/role/recipe/data, no extra
epochs after stage already completed. Each stage exact source/protocol/fit/stop/
query digests and full processed-row counts. No tolerance edits after outcomes.

Fresh-process verifier checks all original byte identities, roles/fit-only LR
stats/golden probabilities, independent MacroF1/formula/gates/epoch choices, fixed
blend on all saved query rows and conditional export. CPU model replay covers32
deterministic proper-role query rows per stage, NOT full neural inference replay.
Same CPU reload probability tolerance1e-6; declared cross-runtime CPU/GPU maxabs
probability tolerance.001; numerical replay never changes quality F1/gain guards.
Independent NumPy softmax/member averaging checked; code confirms per-member CE.
GPU performance/VRAM not claimed from CPU smoke. Return ZIP preserves source/
protocol/best weights/histories/predictions/replay rows; excludes optimizer/full
encoded training cache and raw dataset already present locally.

Qualified export only from best.72430/SHA7a30ea4811dc2c9ff19be654d79b7f85d9706d685a213e1475f310331f0bc51f,
preserving exact id/Tasks1/3/4/5 strings including THB, Swin-tail and Task5mean40%.
No canonical replacement, best-pointer promotion or competition upload. Candidate
public/private unknown until user reports actual score. Never submit smoke fixture.
