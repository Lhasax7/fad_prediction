# Staple or trend piece? Predicting a garment's sales lifespan from its photo

A small model that looks at a clothing photo and predicts whether the item is a **lasting staple** or a **short-lived trend piece**. Labels come from two years of H&M sales lifecycles. At prediction time the model sees only the photo.

**Demo:** [link to 60 to 90 second video]

## Headline result

| | Value |
|---|---|
| Model | Frozen CLIP ViT-L/14 image embeddings + XGBoost |
| Grouped 5-fold CV, AUC | **0.892 +/- 0.011** |
| Product-type-only baseline, AUC | 0.731 +/- 0.018 |
| Lift over baseline | **about 0.16 AUC** |
| Within-product-type AUC (25 types, n >= 40) | median 0.88 |

The model beats a baseline that knows the product type, and it still ranks items correctly inside a single product type (where a type-only baseline is at chance by construction).

## Task and data

- **Data:** Kaggle H&M Personalized Fashion Recommendations (`articles.csv`, `transactions_train.csv`, product images). 31.8M transactions over 105 weeks.
- **Weekly index:** each transaction is assigned a week. Per article I compute `first_wk`, `last_wk`, `active` (weeks with at least one sale), `total` (sales) and `span` (`last_wk - first_wk`).

### Labels (from sales, used only to create labels)

| Class | Rule |
|---|---|
| Staple (0) | span >= 80, active >= 60, total >= 500, still selling in the last 4 weeks |
| Trend piece (1) | launched after week 4, span <= 20, stopped selling before the last 8 weeks, total >= 60 |

The trend thresholds (span <= 20, total >= 60) were chosen from a small sweep so that the two classes came out about the same size (1,783 trend pieces vs 1,788 staples before image filtering). After dropping articles with no image and balancing by downsampling: **3,548 items, 1,774 per class**.

Sales data is only used to build labels, in hindsight. The classifier input is the photo alone, so there is no leakage of sales information into the features.

## Method

1. Encode each photo with frozen `clip-ViT-L-14` (SentenceTransformers), L2-normalised, 768 dimensions.
2. Fit Logistic Regression and XGBoost on the embeddings.
3. Baseline: Logistic Regression on one-hot `product_type_name`.
4. Evaluation always splits by `product_code`, so colour variants of the same product never appear in both train and test.

## Results

### Grouped 5-fold cross-validation (headline)

`StratifiedGroupKFold`, grouped by `product_code`.

| Model | AUC | F1 |
|---|---|---|
| Baseline (product type only) | 0.731 +/- 0.018 | 0.687 +/- 0.029 |
| CLIP + Logistic Regression | 0.870 +/- 0.007 | 0.791 +/- 0.011 |
| CLIP + XGBoost | **0.892 +/- 0.011** | **0.821 +/- 0.018** |
| Ensemble (LR + XGB) | 0.889 +/- 0.009 | 0.816 +/- 0.014 |

Pooled out-of-fold AUC for XGBoost is 0.891. The +/- values are the spread across folds. Training sets overlap, so this is a stability check and not a confidence interval. The ensemble adds nothing over XGBoost because the two models make correlated errors.

### Single split (25% of product codes held out)

| Model | AUC | F1 |
|---|---|---|
| Baseline (product type only) | 0.731 | 0.715 |
| CLIP + LR | 0.853 | 0.760 |
| CLIP + XGBoost | 0.873 | 0.788 |
| Ensemble | 0.870 | 0.787 |

The single split is a secondary result. On a test set of this size, AUC differences under about 0.02 should be treated as noise.

## Robustness

**Volume-controlled check.** Trend pieces sell far less than staples (median 107 vs 1,636 total sales), so the model could in principle just be picking up "looks like a high-volume item." To test this I restricted to items with 200 to 1,000 total sales (525 staples, 365 trend pieces), balanced the classes, and re-ran Logistic Regression over 20 random seeds: **AUC 0.788 +/- 0.031** (about 730 items per run). AUC drops from about 0.87 but stays well above chance. Caveats: the seeds resample the same items, so the spread understates true uncertainty, and only 365 of the 1,774 trend pieces fall in this band, so it covers the best-selling trend pieces.

**Within product type.** Out-of-fold XGBoost predictions, scored per product type with 95% bootstrap intervals (25 types with n >= 40):

| Product type | n | AUC | 95% CI |
|---|---|---|---|
| T-shirt | 240 | 0.911 | 0.872 to 0.948 |
| Shirt | 125 | 0.901 | 0.848 to 0.946 |
| Trousers | 464 | 0.884 | 0.846 to 0.917 |
| Sweater | 314 | 0.841 | 0.790 to 0.888 |
| Dress | 240 | 0.835 | 0.777 to 0.886 |
| Blouse | 146 | 0.812 | 0.743 to 0.878 |
| Cardigan | 57 | 0.638 | 0.473 to 0.791 |
| Underwear Tights | 56 | 0.592 | 0.299 to 0.856 |

Median type AUC is 0.88. 20 of 25 types are at 0.83 or above, and 22 of 25 have intervals entirely above 0.5. The three that do not are Other accessories, Cardigan and Underwear Tights. Do not read the ordering between types as meaningful at these sample sizes. Swimwear bottom scores highest (0.98) but is strongly seasonal, so its label may partly reflect "sold one summer," and I treat it as a best case. Intervals resample rows and not product codes, so they are slightly optimistic.

## Error analysis

I looked at the 12 most confident errors (single split, Logistic Regression).

- **No background or pose clustering.** All photos share a similar grey studio backdrop, and the errors are spread across product types (7 staples predicted as trend, 5 trend pieces predicted as staple).
- **The model reads "visual loudness."** Staples wrongly called trend tend to have a small graphic, print or colour: a sweater with an embroidered heart, a tee with a small logo print, a floral scarf, a lace-collar blouse. Trend pieces wrongly called staple are plain items: basic denim, plain black boxers, plain shorts.
- **Some errors are label noise.** A plain pair of jeans labelled "trend" probably means H&M rotated it out of the catalogue, not that it was a fad. The labels measure sales life, not trendiness.
- **Sanity check on one type.** For Underwear bottom, the items predicted most likely trend were lace or printed, and the items predicted most likely staple were plain black or neutral. That is a sensible split.

## Limitations

- **One retailer.** "Trend piece" here means "short sales life at H&M," not proven trendiness.
- **Catalogue rotation and seasonality** confound the labels. A plain item can be pulled for reasons unrelated to demand, and seasonal items look short-lived.
- **Balanced classes** inflate metrics compared with real-world base rates.
- **Volume confound.** Staples sell about 15x more than trend pieces. The volume-band check controls for this only partly.
- **Small samples per product type**, wide intervals, and many comparisons.
- **The model uses visual loudness as a proxy.** It is a useful signal, not a causal one.
- **Image quality.** A couple of close-up photos appear in the staple underwear group.

## Demo

A Gradio app takes a photo and returns the probability of "trend piece" vs "staple". It uses the same `clip-ViT-L-14` encoder as the training embeddings, followed by the XGBoost model trained on the 75% train split only. Demo images are drawn from the held-out 25%, so the model has never seen them. The headline numbers above come from cross-validation, not from this single demo model.

## Reproduce

Run the notebook top to bottom on Kaggle with the H&M competition data attached (GPU used for CLIP embedding). Embeddings are saved to `X.npy` and item metadata to `data.csv`.

Stack: Python, PyTorch, SentenceTransformers (CLIP), scikit-learn, XGBoost, Gradio.
