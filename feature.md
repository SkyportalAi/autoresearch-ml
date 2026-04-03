# Feature Descriptions — Home Credit Default Risk

## Dataset context

Home Credit loan applications from the unbanked population. 307K applications, 120 features. Target is binary default (8% positive rate). Many applicants have thin/no credit bureau history, so the model must rely on alternative data.

## Core financial features

- **AMT_INCOME_TOTAL**: Applicant's total annual income. No nulls. Wide range ($25K to $100M+ outliers).
- **AMT_CREDIT**: Credit amount of the loan. The total amount to be repaid.
- **AMT_ANNUITY**: Monthly loan annuity — the regular payment amount. Some nulls.
- **AMT_GOODS_PRICE**: Price of the goods being financed. For cash loans this may differ from AMT_CREDIT. Some nulls.

## Loan characteristics

- **NAME_CONTRACT_TYPE**: "Cash loans" or "Revolving loans". Categorical. Cash loans have higher default rates.

## Applicant demographics

- **CODE_GENDER**: "M", "F", or "XNA" (unknown). Categorical.
- **FLAG_OWN_CAR**: "Y" or "N". Categorical.
- **FLAG_OWN_REALTY**: "Y" or "N". Categorical.
- **CNT_CHILDREN**: Number of children. Integer 0-19.
- **CNT_FAM_MEMBERS**: Family size. Float (includes decimals due to encoding).
- **NAME_TYPE_SUITE**: Who accompanied the applicant. Categorical (Unaccompanied, Family, Spouse, etc.). Some nulls.
- **NAME_INCOME_TYPE**: Income source. Categorical (Working, Commercial associate, Pensioner, State servant, etc.).
- **NAME_EDUCATION_TYPE**: Education level. Categorical (Secondary, Higher education, Incomplete higher, etc.).
- **NAME_FAMILY_STATUS**: Marital status. Categorical (Married, Single, Civil marriage, etc.).
- **NAME_HOUSING_TYPE**: Housing situation. Categorical (House/apartment, With parents, Rented, etc.).
- **OCCUPATION_TYPE**: Job type. Categorical with 18 values. ~30% null — missingness itself is informative (may indicate informal/undocumented employment).

## Time-based features (all negative, representing days before application)

- **DAYS_BIRTH**: Applicant's age in negative days (e.g., -10000 ≈ 27 years old). Divide by -365 to get age in years. No nulls.
- **DAYS_EMPLOYED**: Days employed at current job (negative). **CRITICAL: value 365243 is a sentinel meaning NOT employed/retired/unknown — do NOT treat as numeric.** ~18% of rows have this sentinel. All other values are negative.
- **DAYS_REGISTRATION**: Days since registration (negative). How long ago the applicant changed their registration.
- **DAYS_ID_PUBLISH**: Days since ID document was published (negative).
- **DAYS_LAST_PHONE_CHANGE**: Days since last phone change (negative). Some nulls.

## Vehicle

- **OWN_CAR_AGE**: Age of applicant's car in years. Only populated if FLAG_OWN_CAR="Y". ~66% null.

## Contact flags

- **FLAG_MOBIL**: Has mobile phone (1/0). Almost all are 1.
- **FLAG_EMP_PHONE**: Has employer phone (1/0).
- **FLAG_WORK_PHONE**: Has work phone (1/0).
- **FLAG_CONT_MOBILE**: Is mobile phone reachable (1/0).
- **FLAG_PHONE**: Has home phone (1/0).
- **FLAG_EMAIL**: Has email (1/0).

## External credit scores

- **EXT_SOURCE_1**: External score from source 1. Float 0-1. ~56% null. Highly predictive but sparse.
- **EXT_SOURCE_2**: External score from source 2. Float 0-1. ~0.3% null. Most complete external score.
- **EXT_SOURCE_3**: External score from source 3. Float 0-1. ~20% null. Highly predictive.

These three scores are the single most important feature group. They're pre-computed risk scores from external credit bureaus. Missing values are informative — missing scores often indicate thin credit files.

## Region features

- **REGION_POPULATION_RELATIVE**: Population of the region (normalized 0-0.07).
- **REGION_RATING_CLIENT**: Rating of the region (1, 2, or 3). Higher = worse region.
- **REGION_RATING_CLIENT_W_CITY**: Same but weighted by city.
- **REG_REGION_NOT_LIVE_REGION**: 1 if registration region differs from living region.
- **REG_REGION_NOT_WORK_REGION**: 1 if registration region differs from work region.
- **LIVE_REGION_NOT_WORK_REGION**: 1 if living region differs from work region.
- **REG_CITY_NOT_LIVE_CITY**: Same but at city level.
- **REG_CITY_NOT_WORK_CITY**: Same.
- **LIVE_CITY_NOT_WORK_CITY**: Same.

## Housing quality features (~70% null)

These describe the applicant's building/apartment. Each feature comes in three variants:
- **_AVG**: Average for the building
- **_MODE**: Mode for the building
- **_MEDI**: Median for the building

Features: APARTMENTS, BASEMENTAREA, YEARS_BEGINEXPLUATATION, YEARS_BUILD, COMMONAREA, ELEVATORS, ENTRANCES, FLOORSMAX, FLOORSMIN, LANDAREA, LIVINGAPARTMENTS, LIVINGAREA, NONLIVINGAPARTMENTS, NONLIVINGAREA.

Also: **TOTALAREA_MODE** — total area of the dwelling.

**The ~70% missingness is a signal** — applicants without housing data may be in informal housing or unable to provide documentation.

## Housing categorical

- **FONDKAPREMONT_MODE**: Capital repair fund status. ~68% null.
- **HOUSETYPE_MODE**: Type of house. ~70% null.
- **WALLSMATERIAL_MODE**: Wall material. ~70% null.
- **EMERGENCYSTATE_MODE**: Emergency state flag. ~70% null.

## Social circle

- **OBS_30_CNT_SOCIAL_CIRCLE**: Number of observable social contacts (30-day window).
- **DEF_30_CNT_SOCIAL_CIRCLE**: Number of contacts who defaulted (30-day window).
- **OBS_60_CNT_SOCIAL_CIRCLE**: Same for 60-day window.
- **DEF_60_CNT_SOCIAL_CIRCLE**: Same for 60-day window.

## Document flags

- **FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21**: Binary flags (0/1) for whether each document type was provided. 20 document flags. Most are rarely 1. The *count* of documents provided and *which* are missing is more informative than individual flags.

## Credit bureau inquiries

- **AMT_REQ_CREDIT_BUREAU_HOUR**: Number of credit bureau inquiries in the past hour.
- **AMT_REQ_CREDIT_BUREAU_DAY**: Past day.
- **AMT_REQ_CREDIT_BUREAU_WEEK**: Past week.
- **AMT_REQ_CREDIT_BUREAU_MON**: Past month.
- **AMT_REQ_CREDIT_BUREAU_QRT**: Past quarter.
- **AMT_REQ_CREDIT_BUREAU_YEAR**: Past year.

Many recent inquiries (especially HOUR/DAY/WEEK) suggest the applicant is urgently seeking credit — a red flag.

## Other

- **WEEKDAY_APPR_PROCESS_START**: Day of week the application was filed. Categorical.
- **HOUR_APPR_PROCESS_START**: Hour of day (0-23) the application was started.
- **ORGANIZATION_TYPE**: Type of employer organization. Categorical with 57 unique values.
