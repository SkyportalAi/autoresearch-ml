# Program: Credit Card Default Prediction

## Business objective

Predict which credit card customers will default on their next month's payment. The bank's collections team needs to **rank customers by default risk** to prioritize outreach — calling the highest-risk customers first reduces losses. Ranking quality matters more than a hard yes/no threshold.

## Business context and domain knowledge

Credit card default is driven by a combination of financial stress, behavioral patterns, and customer demographics. The dataset contains 6 months of payment history (April-September 2005) for 30,000 Taiwanese credit card holders.

### Key default drivers

1. **Payment delay history**: The single strongest predictor. A customer who has been 2+ months late in recent months is far more likely to default than one who pays on time. Recent delays (PAY_0, the most recent month) matter more than older delays (PAY_6).

2. **Trend in payment behavior**: A customer whose delays are *increasing* month over month (e.g., PAY_6=0, PAY_5=0, PAY_4=1, PAY_3=2) is actively deteriorating. A customer whose delays are *decreasing* is recovering. The slope of the payment delay over 6 months is more predictive than any single month.

3. **Credit utilization**: How much of their credit limit they're using. A customer with BILL_AMT near LIMIT_BAL is maxed out — financially stressed. Low utilization suggests financial headroom.

4. **Repayment ratio**: How much of the bill they actually pay each month (PAY_AMT / BILL_AMT). Customers who pay the minimum or less are high risk. Customers who pay in full are low risk. The ratio matters more than the absolute amounts.

5. **Repayment trend**: Is the customer paying a shrinking percentage of their bill each month? If PAY_AMT/BILL_AMT is declining over the 6 months, they're running out of capacity to pay.

6. **Demographics interact with behavior**: Young, less-educated customers with high utilization are highest risk. But demographics alone are weak predictors — they only matter in combination with payment behavior.

### Known high-risk profiles

- **Spiral Defaulter**: PAY_0 >= 2 (2+ months late now), and delays have been increasing over the past 6 months. They're in a debt spiral.
- **Maxed-Out Minimum Payer**: BILL_AMT close to LIMIT_BAL, and PAY_AMT << BILL_AMT consistently. They're paying minimums on a maxed card.
- **Recent Shock**: Previously good payment history (PAY_3 through PAY_6 all -1 or 0) but recent deterioration (PAY_0 or PAY_2 >= 1). Something changed — job loss, medical event.
- **Chronic Late Payer**: Moderate delays every month (PAY consistently 1-2) but never catches up. Always behind but not yet in full default.

### Feature engineering hints

- **Payment delay trend**: Slope of [PAY_6, PAY_5, PAY_4, PAY_3, PAY_2, PAY_0] — is it rising (getting worse) or falling (improving)?
- **Average delay**: Mean of PAY_0 through PAY_6 captures overall payment discipline
- **Max delay**: The worst single-month delay in the 6-month window
- **Utilization ratios**: BILL_AMT1/LIMIT_BAL (current), BILL_AMT6/LIMIT_BAL (6 months ago)
- **Repayment ratios**: PAY_AMT1/BILL_AMT1 (how much of last bill was paid)
- **Repayment trend**: Slope of [PAY_AMT6/BILL_AMT6, ..., PAY_AMT1/BILL_AMT1]
- **Balance growth**: BILL_AMT1 - BILL_AMT6 (is their balance growing?)
- **Payment consistency**: Std dev of PAY_AMT values (erratic payers are riskier)
- **Months with any delay**: Count of months where PAY > 0
- **SEX, EDUCATION, MARRIAGE are numeric-encoded categories**, not continuous values — treat them as categorical or create meaningful interaction terms

## Dataset

- **source**: `huggingface`
- **hf_dataset**: `scikit-learn/credit-card-clients`
- **bundle_name**: `credit_default`
- **target_column**: `default.payment.next.month`
- **positive_label**: `1`
- **drop_columns**: `ID`

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
