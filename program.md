# Program: Telecom Customer Churn — Proactive Retention Campaign

## Business objective

Predict which customers are most likely to churn so our retention team can run a **proactive outreach campaign**. Marketing has budget to target ~20% of the customer base with tailored retention offers (discounted upgrades, loyalty credits, dedicated support). We need to **rank customers by churn risk** — it's more important to get the ranking right than to get a hard yes/no threshold.

## Business context and domain knowledge

Churn in telecom is NOT random. It clusters around specific customer profiles and lifecycle moments. Years of operational experience have identified these patterns:

### Key churn drivers (in order of importance)

1. **Contract type**: Month-to-month customers churn at 5-8x the rate of contract customers. No cancellation penalty = no friction. But a month-to-month customer who has stayed 24+ months is actually very loyal — they stay by choice.

2. **Tenure lifecycle**: Churn follows a bathtub curve — very high in first 6 months (buyer's remorse, competitor offers), drops through months 12-48 (habitual usage), slight rise at 48+ months (taken-for-granted effect). Surviving the first year is the critical milestone.

3. **Fiber optic paradox**: Fiber customers churn at ~2x the rate of DSL — not because fiber is bad, but because fiber plans cost more, fiber customers are tech-savvy comparison shoppers, and fiber markets have more competitors. DSL customers in rural areas often have no alternative.

4. **Add-on lock-in**: Each add-on service (security, backup, protection, support) creates switching cost. Customers with 0 add-ons are using us as a "dumb pipe" — any cheaper competitor wins instantly. Going from 0 to 1 add-on matters more than 4 to 5.

5. **Payment method signal**: Electronic check customers churn dramatically more than auto-pay customers. Electronic check = active monthly decision to pay = monthly opportunity to cancel. Auto-pay = invisible = inertia favors retention.

6. **Price-to-value ratio**: A customer paying $90/month with 5 add-ons is getting value. A customer paying $80/month with 0 add-ons feels overcharged. The absolute price matters less than what you get for it.

### Known high-risk profiles

- **Unprotected Fiber User**: Fiber optic + no OnlineSecurity + no TechSupport + month-to-month. Premium price, zero lock-in, no support safety net.
- **Solo New Customer**: tenure < 12, no Partner, no Dependents, month-to-month. Zero switching cost, still in "trial period" mentally.
- **Price-Shocked Senior**: SeniorCitizen=1, high MonthlyCharges, electronic check. Price-sensitive, fixed income, active payment decision each month.
- **Feature-Light Customer**: Internet service but zero add-ons across all 6 categories. Using us as a dumb pipe.
- **Passive Downgrader**: Long tenure but current MonthlyCharges significantly higher than historical average (TotalCharges/tenure) — bill increased, resentment building.

### Feature engineering hints

- Count of add-on services (0-6) is more predictive than any individual add-on
- Contract type x tenure interaction captures the loyalty nuance
- MonthlyCharges / (TotalCharges/tenure) reveals recent price changes — ratio > 1 means bill went up
- Electronic check + month-to-month + fiber = triple threat for churn
- The absence of internet service defines a distinct low-churn segment (phone-only customers)

## Dataset

- **source**: `huggingface`
- **hf_dataset**: `scikit-learn/churn-prediction`
- **bundle_name**: `telecom_churn`
- **target_column**: `Churn`
- **positive_label**: `Yes`

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
