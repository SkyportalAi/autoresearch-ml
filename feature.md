# Feature Descriptions — Home Credit Default Risk

## Dataset context

Home Credit loan applications from the unbanked population. 307,511 rows, 122 columns. Binary classification: predict loan default. Class imbalance 11.4:1 (8.07% positive rate) — use stratified CV and class weights.

Source: Kaggle `home-credit-default-risk` competition (`application_train.csv`). Column descriptions below merge the official Kaggle data dictionary, the Kaggle "Special" metadata, and EDA findings from the actual data.

---

## ID and target

- **SK_ID_CURR**: Unique loan ID. int64, no nulls, 307,511 unique. **Drop before modeling** (no predictive value).
- **TARGET**: 1 = client had payment difficulties (late payment > X days on at least one of the first Y installments), 0 = all other cases. int64, no nulls. 8.07% positive rate (24,825 defaults vs 282,686 non-defaults). **Do not create features derived from this column.**

---

## Core financial features

- **AMT_INCOME_TOTAL**: Income of the client. float64, no nulls, 2,548 unique. Range $25,650–$117,000,000. **Extreme right skew (391.6) and kurtosis (191,787)** — log transform or cap recommended. Mean $168,798, median $147,150.
- **AMT_CREDIT**: Credit amount of the loan (total to be repaid). float64, no nulls. Range $45,000–$4,050,000. Mean $599,026.
- **AMT_ANNUITY**: Loan annuity (regular payment amount). float64, 12 nulls (0.004%). Range $1,616–$258,026. Mean $27,108.
- **AMT_GOODS_PRICE**: For consumer loans, the price of the goods for which the loan is given. float64, 278 nulls (0.09%). Range $40,500–$4,050,000. Mean $538,396.

---

## Loan characteristics

- **NAME_CONTRACT_TYPE**: Identification if loan is cash or revolving. Categorical, no nulls. Values: Cash loans (278,232 / 90.5%), Revolving loans (29,279 / 9.5%).

---

## Applicant demographics

- **CODE_GENDER**: Gender of the client. Categorical, no nulls. Values: F (202,448), M (105,059), XNA (4).
- **FLAG_OWN_CAR**: Flag if the client owns a car. Categorical, no nulls. Y/N.
- **FLAG_OWN_REALTY**: Flag if client owns a house or flat. Categorical, no nulls. Y/N.
- **CNT_CHILDREN**: Number of children the client has. int64, no nulls. Range 0–19. 70% have 0 children.
- **CNT_FAM_MEMBERS**: How many family members the client has. float64 (stored as float but all values are whole numbers), 2 nulls. Range 1–20. 51.5% = 2.
- **NAME_TYPE_SUITE**: Who was accompanying client when applying for the loan. Categorical, 1,292 nulls (0.42%). 7 values. Top: Unaccompanied (248,526), Family (40,149).
- **NAME_INCOME_TYPE**: Client's income type. Categorical, no nulls. 8 values. Top: Working (158,774), Commercial associate (71,617), Pensioner (55,362).
- **NAME_EDUCATION_TYPE**: Level of highest education the client achieved. Categorical, no nulls. 5 values. Top: Secondary/secondary special (218,391), Higher education (74,863).
- **NAME_FAMILY_STATUS**: Family status of the client. Categorical, no nulls. 6 values. Top: Married (196,432), Single (45,444).
- **NAME_HOUSING_TYPE**: Housing situation of the client. Categorical, no nulls. 6 values. Top: House/apartment (272,868), With parents (14,840).
- **OCCUPATION_TYPE**: What kind of occupation the client has. Categorical, **96,391 nulls (31.35%)** — missingness is informative (may indicate informal/undocumented/gig employment). 18 unique values. Top: Laborers (55,186), Sales staff (32,102).
- **ORGANIZATION_TYPE**: Type of organization where client works. Categorical, no nulls. **58 unique values** (high cardinality — frequency-encode). Top: Business Entity Type 3 (67,992), XNA (55,374 — same count as DAYS_EMPLOYED sentinel, likely the same population), Self-employed (38,412).

---

## Time-based features

**All negative values, representing days before application. Kaggle Special: "time only relative to the application" — these are not absolute dates. Important for deployment: values become stale over time.**

- **DAYS_BIRTH**: Client's age in days at the time of application. int64, no nulls. Range -25,229 to -7,489. Divide by -365 to get age in years (~20–69 years).
- **DAYS_EMPLOYED**: How many days before the application the person started current employment. int64, no nulls. Range -17,912 to 365,243. **CRITICAL SENTINEL: value 365,243 appears in 55,374 rows (18.01%) — means NOT employed/retired/unknown. Replace with NaN and create a binary flag `employed_flag`.** All other values are negative.
- **DAYS_REGISTRATION**: How many days before the application the client changed their registration. float64, no nulls. Range -24,672 to 0. Mean -4,986.
- **DAYS_ID_PUBLISH**: How many days before the application the client changed the identity document used for the loan. int64, no nulls. Range -7,197 to 0. Mean -2,994.
- **DAYS_LAST_PHONE_CHANGE**: How many days before application did client change phone. float64, 1 null. Range -4,292 to 0. Mean -963.

---

## Vehicle

- **OWN_CAR_AGE**: Age of client's car in years. float64, **202,929 nulls (65.99%)** — only populated if FLAG_OWN_CAR="Y". Range 0–91. Mean 12.06.

---

## Contact flags

Binary (0/1) indicators for whether the client provided various contact methods.

- **FLAG_MOBIL**: Did client provide mobile phone. **99.997% = 1 — effectively constant. DROP this feature.**
- **FLAG_EMP_PHONE**: Did client provide work phone. Mean 0.82.
- **FLAG_WORK_PHONE**: Did client provide home phone (Kaggle labels this "home phone"). Mean 0.20.
- **FLAG_CONT_MOBILE**: Was mobile phone reachable. **99.81% = 1 — near-constant, consider dropping.** Mean 0.998.
- **FLAG_PHONE**: Did client provide home phone. Mean 0.28.
- **FLAG_EMAIL**: Did client provide email. Mean 0.057.

---

## External credit scores

**Kaggle Special: "normalized". These are pre-computed risk scores from external credit bureaus, normalized to [0, 1]. The single most important feature group.**

- **EXT_SOURCE_1**: Normalized score from external data source. float64, **173,378 nulls (56.38%)**. Range 0.015–0.963. Highly predictive but sparse.
- **EXT_SOURCE_2**: Normalized score from external data source. float64, **660 nulls (0.21%)**. Range ~0–0.855. Most complete external score.
- **EXT_SOURCE_3**: Normalized score from external data source. float64, **60,965 nulls (19.83%)**. Range 0.0005–0.896. Highly predictive.

Missing values are informative — missing scores often indicate thin credit files. Create missingness flags for all three.

---

## Region features

- **REGION_POPULATION_RELATIVE**: Normalized population of region where client lives (higher = more populated). float64, no nulls. Kaggle Special: "normalized". Range 0.0003–0.073.
- **REGION_RATING_CLIENT**: Our rating of the region where client lives (1, 2, or 3). int64, no nulls. Higher = worse region. 73.8% = 2 (low variance).
- **REGION_RATING_CLIENT_W_CITY**: Same rating but weighted by city (1, 2, or 3). int64, no nulls. 74.6% = 2.

### Address mismatch flags (binary 0/1)

These flag whether the client's registration, living, and work addresses differ. All int64, no nulls.

| Column | Mean | Notes |
|--------|------|-------|
| REG_REGION_NOT_LIVE_REGION | 0.015 | **98.5% zeros** — very sparse |
| REG_REGION_NOT_WORK_REGION | 0.051 | |
| LIVE_REGION_NOT_WORK_REGION | 0.041 | |
| REG_CITY_NOT_LIVE_CITY | 0.078 | |
| REG_CITY_NOT_WORK_CITY | 0.231 | Most informative of the group |
| LIVE_CITY_NOT_WORK_CITY | 0.180 | |

Consider a composite "address_mismatch_count" summing all 6 flags.

---

## Housing quality features

**Kaggle Special: "normalized". Normalized information about the building where the client lives.**

Each base feature comes in three variants:
- **_AVG**: Average for the building
- **_MODE**: Mode for the building
- **_MEDI**: Median for the building

**CRITICAL: The three variants correlate >0.999 (near-perfect multicollinearity). Keep only ONE variant per base feature (recommend _MEDI). This reduces 42 numeric columns to 14.**

### Numeric housing features (14 base features x 3 variants = 42 columns)

Full column names: APARTMENTS_AVG, APARTMENTS_MODE, APARTMENTS_MEDI, BASEMENTAREA_AVG, BASEMENTAREA_MODE, BASEMENTAREA_MEDI, YEARS_BEGINEXPLUATATION_AVG, YEARS_BEGINEXPLUATATION_MODE, YEARS_BEGINEXPLUATATION_MEDI, YEARS_BUILD_AVG, YEARS_BUILD_MODE, YEARS_BUILD_MEDI, COMMONAREA_AVG, COMMONAREA_MODE, COMMONAREA_MEDI, ELEVATORS_AVG, ELEVATORS_MODE, ELEVATORS_MEDI, ENTRANCES_AVG, ENTRANCES_MODE, ENTRANCES_MEDI, FLOORSMAX_AVG, FLOORSMAX_MODE, FLOORSMAX_MEDI, FLOORSMIN_AVG, FLOORSMIN_MODE, FLOORSMIN_MEDI, LANDAREA_AVG, LANDAREA_MODE, LANDAREA_MEDI, LIVINGAPARTMENTS_AVG, LIVINGAPARTMENTS_MODE, LIVINGAPARTMENTS_MEDI, LIVINGAREA_AVG, LIVINGAREA_MODE, LIVINGAREA_MEDI, NONLIVINGAPARTMENTS_AVG, NONLIVINGAPARTMENTS_MODE, NONLIVINGAPARTMENTS_MEDI, NONLIVINGAREA_AVG, NONLIVINGAREA_MODE, NONLIVINGAREA_MEDI.

| Base feature | Null % | Mean (_MEDI) | Notes |
|-------------|--------|-------------|-------|
| APARTMENTS | 50.75% | 0.118 | Apartment size |
| BASEMENTAREA | 58.52% | 0.088 | Basement area |
| YEARS_BEGINEXPLUATATION | 48.78% | 0.978 | Age of building (exploitation start) |
| YEARS_BUILD | 66.50% | 0.756 | Year of building construction |
| COMMONAREA | 69.87% | 0.045 | Common area |
| ELEVATORS | 53.30% | 0.078 | Number of elevators |
| ENTRANCES | 50.35% | 0.149 | Number of entrances. **Sentinel-like: 22-24% = 0.1379** |
| FLOORSMAX | 49.76% | 0.226 | Max floors. **Sentinel-like: 40-42% = 0.1667** |
| FLOORSMIN | 67.85% | 0.232 | Min floors. **Sentinel-like: 33-35% = 0.2083** |
| LANDAREA | 59.38% | 0.067 | Land area |
| LIVINGAPARTMENTS | 68.35% | 0.102 | Living apartments |
| LIVINGAREA | 50.19% | 0.109 | Living area |
| NONLIVINGAPARTMENTS | 69.43% | 0.009 | Non-living apartments. ~85-91% in top 5 values |
| NONLIVINGAREA | 55.18% | 0.028 | Non-living area |

**Also: TOTALAREA_MODE** — total area of the dwelling. float64, 48.27% null. Mean 0.103. MODE variant only, no AVG/MEDI.

**The 49-70% missingness is a signal** — applicants without housing data may be in informal housing or unable to provide documentation. Create a binary `has_housing_data` flag.

**Suspicious high-frequency constants** in FLOORSMAX, FLOORSMIN, ENTRANCES suggest default/imputed values rather than real data. Consider treating them as NaN and creating sentinel flags.

### Categorical housing features (MODE variant only)

- **FONDKAPREMONT_MODE**: Capital repair fund status. **68.39% null**. 4 values: reg oper account (73,830), reg oper spec account (12,080), not specified (5,687), org spec account (4,403).
- **HOUSETYPE_MODE**: Type of house. **50.18% null**. 3 values: block of flats (150,503), specific housing (1,499), terraced house (1,212).
- **WALLSMATERIAL_MODE**: Wall material. **50.84% null**. 7 values: Panel (66,040), Stone/brick (64,815), Block (9,253).
- **EMERGENCYSTATE_MODE**: Emergency state flag. **47.40% null**. 2 values: No (159,428), Yes (2,328).

---

## Social circle

- **OBS_30_CNT_SOCIAL_CIRCLE**: How many observations of client's social surroundings with observable 30 DPD (days past due) default. float64, 1,021 nulls (0.33%). Range 0–348. **Extreme kurtosis (1,425)** — heavy-tailed. Mean 1.42.
- **DEF_30_CNT_SOCIAL_CIRCLE**: How many of client's social surroundings defaulted on 30 DPD. float64, 1,021 nulls (0.33%). Range 0–34. Mean 0.14.
- **OBS_60_CNT_SOCIAL_CIRCLE**: Same for 60 DPD. float64, 1,021 nulls (0.33%). Range 0–344. **Extreme kurtosis (1,410)**. Mean 1.41.
- **DEF_60_CNT_SOCIAL_CIRCLE**: Same for 60 DPD defaults. float64, 1,021 nulls (0.33%). Range 0–24. Mean 0.10.

---

## Document flags

Binary flags (0/1) for whether each document type was provided. All int64, no nulls.

| Column | Mean (≈ % provided) | Notes |
|--------|---------------------|-------|
| FLAG_DOCUMENT_2 | 0.004% | **Near-zero — drop** |
| FLAG_DOCUMENT_3 | 71.0% | Most commonly provided document |
| FLAG_DOCUMENT_4 | 0.008% | **Near-zero — drop** |
| FLAG_DOCUMENT_5 | 1.5% | |
| FLAG_DOCUMENT_6 | 8.8% | |
| FLAG_DOCUMENT_7 | 0.02% | **Near-zero — drop** |
| FLAG_DOCUMENT_8 | 8.1% | |
| FLAG_DOCUMENT_9 | 0.39% | |
| FLAG_DOCUMENT_10 | 0.002% | **Near-zero — drop** |
| FLAG_DOCUMENT_11 | 0.39% | |
| FLAG_DOCUMENT_12 | 0.0007% | **Near-zero — drop** |
| FLAG_DOCUMENT_13 | 0.35% | |
| FLAG_DOCUMENT_14 | 0.29% | |
| FLAG_DOCUMENT_15 | 0.12% | |
| FLAG_DOCUMENT_16 | 0.99% | |
| FLAG_DOCUMENT_17 | 0.027% | **Near-zero — drop** |
| FLAG_DOCUMENT_18 | 0.81% | |
| FLAG_DOCUMENT_19 | 0.06% | |
| FLAG_DOCUMENT_20 | 0.05% | |
| FLAG_DOCUMENT_21 | 0.03% | |

**Recommendation**: Drop FLAG_DOCUMENT_2, 4, 7, 10, 12, 17 (all <0.03% — essentially constant zero). Create a composite `documents_provided_count` summing the remaining flags. The *count* and *which* documents are missing is more informative than individual flags.

---

## Credit bureau inquiries

**IMPORTANT: These are NON-OVERLAPPING sequential time windows (each excludes the previous tier). Summing all 6 gives total inquiries in the past year.**

All float64, all 41,519 nulls (13.50%).

| Column | Kaggle description | Mean | Notes |
|--------|-------------------|------|-------|
| AMT_REQ_CREDIT_BUREAU_HOUR | Inquiries in past hour | 0.006 | **99.4% zeros** |
| AMT_REQ_CREDIT_BUREAU_DAY | Inquiries in past day (**excluding** past hour) | 0.007 | **99.4% zeros** |
| AMT_REQ_CREDIT_BUREAU_WEEK | Inquiries in past week (**excluding** past day) | 0.034 | **96.8% zeros** |
| AMT_REQ_CREDIT_BUREAU_MON | Inquiries in past month (**excluding** past week) | 0.267 | |
| AMT_REQ_CREDIT_BUREAU_QRT | Inquiries in past 3 months (**excluding** past month) | 0.266 | Max 261 (outlier) |
| AMT_REQ_CREDIT_BUREAU_YEAR | Inquiries in past year (**excluding** past 3 months) | 1.900 | |

Many recent inquiries (especially HOUR/DAY/WEEK) suggest urgent credit-seeking — a red flag. Consider creating: `total_credit_inquiries` (sum of all 6), `recent_inquiry_flag` (any inquiry in HOUR/DAY/WEEK > 0).

---

## Application timing

- **WEEKDAY_APPR_PROCESS_START**: On which day of the week the client applied for the loan. Categorical, no nulls. 7 values, roughly uniform (Tuesday slightly highest at 17.5%).
- **HOUR_APPR_PROCESS_START**: Approximately at what hour the client applied. int64, no nulls. Range 0–23. Kaggle Special: "rounded" (approximate, not exact). Mean 12.06.

---

## Feature engineering hints (data audit summary)

1. **Drop constant/near-constant columns**: FLAG_MOBIL, FLAG_DOCUMENT_2/4/7/10/12/17
2. **Handle sentinels**: DAYS_EMPLOYED (365243 → NaN + flag), housing features FLOORSMAX/FLOORSMIN/ENTRANCES high-frequency constants
3. **Log-transform**: AMT_INCOME_TOTAL (skew 391.6), OBS_30/60_CNT_SOCIAL_CIRCLE (kurtosis >1,400)
4. **Deduplicate housing**: Pick one of _AVG/_MEDI/_MODE per base feature (>0.999 correlation)
5. **Frequency-encode**: ORGANIZATION_TYPE (58 values), OCCUPATION_TYPE (18 values)
6. **Missingness flags**: EXT_SOURCE_1 (56%), OCCUPATION_TYPE (31%), housing features (49-70%), OWN_CAR_AGE (66%)
7. **Composite features**: documents_provided_count, address_mismatch_count, total_credit_inquiries, has_housing_data
8. **Ratios**: AMT_CREDIT / AMT_INCOME_TOTAL (debt-to-income), AMT_ANNUITY / AMT_INCOME_TOTAL (payment burden), AMT_GOODS_PRICE / AMT_CREDIT (loan markup)
