# Program: Home Credit Default Risk

## Business objective

Predict which loan applicants will default on their loan. Home Credit serves the unbanked population — people with little or no traditional credit history. The goal is to **rank applicants by default risk** so the lender can approve more loans to creditworthy people while managing risk. Ranking quality (average_precision) matters more than a binary cutoff because the business adjusts interest rates and loan sizes based on risk tier.

## Business context and domain knowledge

### Why this is hard

The applicants often have thin or nonexistent credit bureau files. Traditional credit scores don't work. The lender must infer creditworthiness from alternative data: employment history, housing situation, social connections, and the application itself.

### Key default drivers

1. **External credit scores (EXT_SOURCE_1/2/3)**: These are scores from external data sources — the single most predictive features. They're pre-computed risk scores, but they have significant missing values. The *combination* and *agreement* between scores matters (all three low = very high risk; mixed signals = moderate risk).

2. **Debt burden**: The ratio of the loan annuity (AMT_ANNUITY) to income (AMT_INCOME_TOTAL) is critical. Also: credit amount relative to goods price (how much is financed vs. the actual value), and credit amount relative to income.

3. **Employment stability**: DAYS_EMPLOYED measures how long the applicant has been at their current job (negative = days before application). Very short employment or anomalous values (365243 = unemployed/retired, a known sentinel) are strong signals.

4. **Age and life stage**: DAYS_BIRTH (negative, days before application). Younger applicants default more. The interaction between age and loan amount matters — young person taking a large loan = high risk.

5. **Document completeness**: FLAG_DOCUMENT_* columns indicate which documents were provided. The *number* of documents submitted and *which specific ones* are missing correlate with default risk. Missing documents may indicate inability to provide proof of income/employment.

6. **Housing information**: The APARTMENTS/LIVINGAREA/BASEMENTAREA columns (with _AVG/_MODE/_MEDI suffixes) describe the applicant's housing. Heavy missingness (~70%) is itself a signal — missing housing data correlates with higher default rates.

7. **Social circle defaults**: DEF_30_CNT_SOCIAL_CIRCLE and DEF_60_CNT_SOCIAL_CIRCLE count how many people in the applicant's social circle have defaulted. Social risk contagion is real.

8. **Credit bureau inquiries**: AMT_REQ_CREDIT_BUREAU_* columns count how many times the applicant's credit was checked. Many recent inquiries (HOUR, DAY, WEEK) suggest desperation for credit — a red flag.

### Known high-risk profiles

- **Young, low-income, high credit**: DAYS_BIRTH > -10000 (under ~27), AMT_INCOME_TOTAL < median, AMT_CREDIT > 2x income. Over-leveraged young borrower.
- **Unstable employment, no car, no realty**: DAYS_EMPLOYED > -365 (less than 1 year), FLAG_OWN_CAR=N, FLAG_OWN_REALTY=N. No asset backing, short employment.
- **Low external scores with missing data**: EXT_SOURCE_2 < 0.3, EXT_SOURCE_3 missing. Thin credit file with bad scores.
- **High bureau inquiries + cash loan**: AMT_REQ_CREDIT_BUREAU_MON > 3, NAME_CONTRACT_TYPE=Cash loans. Actively seeking credit from multiple lenders.
- **Social circle with defaults**: DEF_30_CNT_SOCIAL_CIRCLE > 0. People they know have defaulted — social risk factor.

### Feature engineering hints

- **Debt ratios**: AMT_ANNUITY/AMT_INCOME_TOTAL (debt-to-income), AMT_CREDIT/AMT_GOODS_PRICE (loan-to-value), AMT_CREDIT/AMT_INCOME_TOTAL (credit-to-income)
- **External score combinations**: mean of available EXT_SOURCE, count of missing EXT_SOURCE, product of EXT_SOURCE scores, max-min spread
- **Age and employment derived**: age in years from DAYS_BIRTH, employment years from DAYS_EMPLOYED, employment-to-age ratio (how much of their life they've worked), flag for DAYS_EMPLOYED=365243 sentinel
- **Document completeness score**: count of FLAG_DOCUMENT_* that are 1
- **Housing data completeness**: count of non-null housing columns (APARTMENTS_AVG, LIVINGAREA_AVG, etc.)
- **Credit bureau pressure**: sum of recent inquiries (HOUR + DAY + WEEK), ratio of recent to yearly
- **Income per family member**: AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
- **Annuity burden per family**: AMT_ANNUITY / CNT_FAM_MEMBERS
- **Region risk**: REGION_RATING_CLIENT interaction with income and credit amount

## Dataset

- **source**: `csv`
- **csv_path**: `/tmp/home_credit/application_train.csv`
- **bundle_name**: `home_credit`
- **target_column**: `TARGET`
- **positive_label**: `1`
- **drop_columns**: `SK_ID_CURR` (application ID, no predictive value)

## Model families to evaluate

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `xgboost`
- `lightgbm`
- `catboost`

## Primary metric

`average_precision`

## Search constraints

- **max_total_trials**: `20`
- **min_trials_per_family**: `3`
- **max_trials_per_family**: `8`
- **max_consecutive_non_improvements**: `5`
- **min_improvement**: `0.001`
- **per_run_timeout_minutes**: `10`

## Hardware

- **preferred_backend**: `auto`
