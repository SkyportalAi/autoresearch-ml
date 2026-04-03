# Feature Descriptions — Credit Card Default Prediction

## Dataset context

Data from a Taiwanese bank covering 30,000 credit card clients, April-September 2005. Each row is one customer. Features include 6 months of payment history, bill statements, and demographics.

## Columns

- **LIMIT_BAL**: Credit limit in NT dollars (New Taiwan Dollar). Ranges from 10,000 to 1,000,000. Higher limits generally indicate the bank trusts the customer more. No nulls.

- **SEX**: Numeric-encoded category. 1 = Male, 2 = Female. This is NOT a continuous variable — do not treat the numeric value as meaningful (2 is not "more" than 1). No nulls.

- **EDUCATION**: Numeric-encoded category. 1 = Graduate school, 2 = University, 3 = High school, 4 = Others, 5 = Unknown, 6 = Unknown. Values 0, 5, 6 appear in the data and are unlabeled/unknown. This is NOT continuous — a value of 3 is not "1.5x more educated" than 2. No nulls.

- **MARRIAGE**: Numeric-encoded category. 1 = Married, 2 = Single, 3 = Others. Value 0 also appears and is unlabeled. This is NOT continuous. No nulls.

- **AGE**: Integer, customer's age in years. Ranges roughly 21-79. This IS continuous. No nulls.

- **PAY_0**: Payment status in the most recent month (September 2005). Values: -2 = no consumption, -1 = paid in full, 0 = revolving credit (minimum payment made), 1 = payment delay of 1 month, 2 = payment delay of 2 months, ..., 8 = payment delay of 8 months. Despite being named PAY_0, this is the MOST RECENT month. Higher values = worse payment behavior. This is the single most predictive raw feature.

- **PAY_2**: Payment status in August 2005 (second most recent). Same encoding as PAY_0. Note: there is no PAY_1 in the dataset — the column naming skips from PAY_0 to PAY_2.

- **PAY_3**: Payment status in July 2005. Same encoding.

- **PAY_4**: Payment status in June 2005. Same encoding.

- **PAY_5**: Payment status in May 2005. Same encoding.

- **PAY_6**: Payment status in April 2005 (oldest month). Same encoding.

- **BILL_AMT1**: Bill statement amount in September 2005 (most recent) in NT dollars. Can be negative (credit balance / overpayment). No nulls.

- **BILL_AMT2**: Bill statement amount in August 2005. Same encoding.

- **BILL_AMT3**: Bill statement amount in July 2005.

- **BILL_AMT4**: Bill statement amount in June 2005.

- **BILL_AMT5**: Bill statement amount in May 2005.

- **BILL_AMT6**: Bill statement amount in April 2005 (oldest).

- **PAY_AMT1**: Amount paid in September 2005 in NT dollars. This is how much the customer actually paid toward their bill. Ranges from 0 to very large values. No nulls.

- **PAY_AMT2**: Amount paid in August 2005.

- **PAY_AMT3**: Amount paid in July 2005.

- **PAY_AMT4**: Amount paid in June 2005.

- **PAY_AMT5**: Amount paid in May 2005.

- **PAY_AMT6**: Amount paid in April 2005 (oldest).

- **default.payment.next.month**: Target variable. 1 = customer defaulted on their October 2005 payment, 0 = did not default. ~22% positive rate. Do NOT derive features from this column.

## Temporal structure

The data has a clear time dimension across the 6 months:

```
Oldest                                              Most recent
PAY_6 → PAY_5 → PAY_4 → PAY_3 → PAY_2 → PAY_0
BILL_AMT6 → BILL_AMT5 → BILL_AMT4 → BILL_AMT3 → BILL_AMT2 → BILL_AMT1
PAY_AMT6 → PAY_AMT5 → PAY_AMT4 → PAY_AMT3 → PAY_AMT2 → PAY_AMT1
```

The suffix numbers are confusing: AMT1 is the most recent, AMT6 is the oldest. PAY_0 is the most recent, PAY_6 is the oldest. PAY_1 does not exist.

## Special value notes

- PAY values of -2 and -1 are "good" (no consumption / paid in full). 0 is "okay" (revolving/minimum). 1+ is "bad" (delayed).
- BILL_AMT can be negative when the customer has a credit balance (overpaid).
- PAY_AMT of 0 means the customer made no payment that month — very concerning if combined with a positive BILL_AMT.
