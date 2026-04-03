# Feature Descriptions — Telecom Customer Churn

## Columns

- **customerID**: Unique alphanumeric identifier per customer. No predictive value — drop this.

- **gender**: Male / Female. No nulls.

- **SeniorCitizen**: 0 = not senior, 1 = senior citizen (65+). Encoded as integer, not Yes/No like other binary columns.

- **Partner**: Yes / No. Whether customer lives with a partner.

- **Dependents**: Yes / No. Whether customer has dependents (children, elderly family on the plan).

- **tenure**: Integer, 0–72. Number of months the customer has been with the company. A value of 0 means the customer signed up this month. No nulls.

- **PhoneService**: Yes / No. Whether customer has a phone line.

- **MultipleLines**: Yes / No / "No phone service". The third value means the customer has no phone service at all — it's NOT a missing value, it's a distinct segment (internet-only customers).

- **InternetService**: DSL / Fiber optic / No. "No" means the customer is phone-only. Fiber optic is the premium tier with higher monthly costs.

- **OnlineSecurity**: Yes / No / "No internet service". Add-on security product. "No internet service" means the customer has no internet — distinct from "No" (has internet but didn't subscribe to security).

- **OnlineBackup**: Yes / No / "No internet service". Cloud backup add-on. Same encoding as OnlineSecurity.

- **DeviceProtection**: Yes / No / "No internet service". Device insurance add-on. Same encoding.

- **TechSupport**: Yes / No / "No internet service". Premium tech support add-on. Same encoding.

- **StreamingTV**: Yes / No / "No internet service". TV streaming add-on. Same encoding.

- **StreamingMovies**: Yes / No / "No internet service". Movie streaming add-on. Same encoding.

- **Contract**: Month-to-month / One year / Two year. The customer's contract type. Month-to-month customers can cancel anytime with no penalty.

- **PaperlessBilling**: Yes / No. Whether the customer receives bills electronically.

- **PaymentMethod**: Four values — "Electronic check" / "Mailed check" / "Bank transfer (automatic)" / "Credit card (automatic)". The "(automatic)" suffix means auto-pay is enabled.

- **MonthlyCharges**: Float, 18.25–118.75. Current monthly bill in dollars. No nulls.

- **TotalCharges**: **Stored as string, not numeric.** Must be converted with `pd.to_numeric(errors='coerce')`. Empty strings exist for tenure=0 customers (they haven't been billed yet) — these become NaN on conversion and should be filled with 0.0. Represents cumulative amount paid over the customer's lifetime.

- **Churn**: Target variable. 1 = customer left, 0 = customer stayed. ~27% positive rate. Do NOT derive features from this column.
