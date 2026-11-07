# ============================================================
# OLIST E-COMMERCE SALES ANALYSIS
# File: analysis.py
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Supports both data/ and data/raw/ folder structures.
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)

ANALYSIS_YEAR = 2017

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def find_csv(filename):
    """
    Search for a CSV file inside the data folder and its
    subfolders. This supports data/ and data/raw/ structures.
    """

    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATA_DIR}"
        )

    return matches[0]


def load_dataset(filename):
    path = find_csv(filename)
    print(f"Loading: {filename}")
    return pd.read_csv(path)


orders = load_dataset("olist_orders_dataset.csv")
items = load_dataset("olist_order_items_dataset.csv")
products = load_dataset("olist_products_dataset.csv")
customers = load_dataset("olist_customers_dataset.csv")
reviews = load_dataset("olist_order_reviews_dataset.csv")
category_translation = load_dataset(
    "product_category_name_translation.csv"
)


# ============================================================
# 3. DATA CLEANING - REMOVE DUPLICATES FIRST
# ============================================================

print("\n========== DATA CLEANING ==========")

cleaning_log = []


def log_cleaning(dataset_name, before, after, action):
    removed = before - after

    cleaning_log.append({
        "Dataset": dataset_name,
        "Rows Before": before,
        "Rows After": after,
        "Rows Removed": removed,
        "Action": action
    })


# Orders: one record per order_id
before = len(orders)
orders = orders.drop_duplicates(subset=["order_id"])
log_cleaning(
    "Orders", before, len(orders),
    "Removed duplicate order IDs"
)


# Order items: one record per order item
before = len(items)
items = items.drop_duplicates(
    subset=["order_id", "order_item_id"]
)
log_cleaning(
    "Order Items", before, len(items),
    "Removed duplicate order items"
)


# Products: one record per product
before = len(products)
products = products.drop_duplicates(subset=["product_id"])
log_cleaning(
    "Products", before, len(products),
    "Removed duplicate product IDs"
)


# Customers: one record per customer
before = len(customers)
customers = customers.drop_duplicates(subset=["customer_id"])
log_cleaning(
    "Customers", before, len(customers),
    "Removed duplicate customer IDs"
)


# Reviews: remove exact duplicate rows
before = len(reviews)
reviews = reviews.drop_duplicates()
log_cleaning(
    "Reviews", before, len(reviews),
    "Removed exact duplicate review rows"
)


# Category translation: one translation per category
before = len(category_translation)
category_translation = category_translation.drop_duplicates(
    subset=["product_category_name"]
)
log_cleaning(
    "Category Translation", before, len(category_translation),
    "Removed duplicate category translations"
)


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    if column in orders.columns:
        orders[column] = pd.to_datetime(
            orders[column], errors="coerce"
        )


for column in ["price", "freight_value"]:
    if column in items.columns:
        items[column] = pd.to_numeric(
            items[column], errors="coerce"
        )


reviews["review_score"] = pd.to_numeric(
    reviews["review_score"], errors="coerce"
)


# ============================================================
# 5. TRANSLATE PRODUCT CATEGORIES INTO ENGLISH
# ============================================================

products = products.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

products["product_category_name_english"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
    .fillna("unknown")
)

products["product_category_name_english"] = (
    products["product_category_name_english"]
    .replace("", "unknown")
)


# ============================================================
# 6. MERGE CUSTOMER INFORMATION
# ============================================================

customer_columns = [
    "customer_id",
    "customer_state",
    "customer_city"
]

customers_small = customers[
    [column for column in customer_columns if column in customers.columns]
].copy()


# ============================================================
# 7. MERGE PRODUCT INFORMATION WITH ORDER ITEMS
# ============================================================

fact = items.merge(
    products[
        [
            "product_id",
            "product_category_name_english"
        ]
    ],
    on="product_id",
    how="left"
)

fact = fact.merge(
    orders[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp"
        ]
    ],
    on="order_id",
    how="left"
)

fact = fact.merge(
    customers_small,
    on="customer_id",
    how="left"
)

fact["product_category_name_english"] = (
    fact["product_category_name_english"]
    .fillna("unknown")
)

fact["customer_state"] = (
    fact["customer_state"]
    .fillna("unknown")
)

fact["seller_id"] = fact["seller_id"].fillna("unknown")


# ============================================================
# 8. HANDLE INVALID AND MISSING NUMERICAL VALUES
# ============================================================

# Convert negative prices and freight values into missing values.
fact.loc[fact["price"] < 0, "price"] = np.nan
fact.loc[fact["freight_value"] < 0, "freight_value"] = np.nan


# ---------- PRICE IMPUTATION ----------

# First: fill missing price using the same product's mean.
product_price_mean = fact.groupby("product_id")["price"].transform("mean")

fact["price"] = fact["price"].fillna(product_price_mean)


# Second: fill remaining missing price using category mean.
category_price_mean = fact.groupby(
    "product_category_name_english"
)["price"].transform("mean")

fact["price"] = fact["price"].fillna(category_price_mean)


# Third: use the overall median only for any remaining missing values.
# This is a final fallback, not the main imputation method.
overall_price_median = fact["price"].median()

fact["price"] = fact["price"].fillna(overall_price_median)


# ---------- FREIGHT IMPUTATION ----------

# First: fill missing freight using the customer's state mean.
state_freight_mean = fact.groupby(
    "customer_state"
)["freight_value"].transform("mean")

fact["freight_value"] = fact["freight_value"].fillna(
    state_freight_mean
)


# Second: fill remaining missing freight using seller mean.
seller_freight_mean = fact.groupby(
    "seller_id"
)["freight_value"].transform("mean")

fact["freight_value"] = fact["freight_value"].fillna(
    seller_freight_mean
)


# Final fallback for any remaining missing freight values.
overall_freight_median = fact["freight_value"].median()

fact["freight_value"] = fact["freight_value"].fillna(
    overall_freight_median
)


# Remove rows where price or freight is still unavailable.
fact = fact.dropna(subset=["price", "freight_value"])


# ============================================================
# 9. HANDLE REVIEWS
# ============================================================

# A single order can have multiple review records.
# Calculate the average review score for each order.
order_reviews = (
    reviews.groupby("order_id", as_index=False)
    ["review_score"]
    .mean()
)


# ============================================================
# 10. FILTER VALID ORDERS AND ANALYSIS YEAR
# ============================================================

# Exclude orders that do not represent completed sales.
excluded_statuses = ["canceled", "unavailable"]

valid_orders = orders[
    ~orders["order_status"].isin(excluded_statuses)
].copy()


# Keep orders with a valid purchase date.
valid_orders = valid_orders.dropna(
    subset=["order_purchase_timestamp"]
)


# Filter orders to the selected analysis year.
valid_orders = valid_orders[
    valid_orders["order_purchase_timestamp"].dt.year
    == ANALYSIS_YEAR
].copy()


# Keep only items belonging to valid orders.
fact = fact[
    fact["order_id"].isin(valid_orders["order_id"])
].copy()


# ============================================================
# 11. CREATE CLEANED ORDER-ITEM DATASET
# ============================================================

fact["revenue"] = fact["price"]

fact["total_item_value"] = (
    fact["price"] + fact["freight_value"]
)

fact["order_month"] = (
    fact["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

fact["month_name"] = (
    fact["order_purchase_timestamp"].dt.strftime("%B")
)

fact["year"] = (
    fact["order_purchase_timestamp"].dt.year
)

fact.to_csv(
    OUTPUT_DIR / "cleaned_order_items.csv",
    index=False
)


# ============================================================
# 12. CREATE ORDER-LEVEL DATASET
# ============================================================

# Important:
# We calculate order-level values separately to avoid
# counting the same order multiple times.

order_sales = (
    fact.groupby("order_id", as_index=False)
    .agg(
        product_revenue=("price", "sum"),
        freight_total=("freight_value", "sum"),
        total_items=("order_item_id", "count"),
        customer_state=("customer_state", "first"),
        order_purchase_timestamp=("order_purchase_timestamp", "first")
    )
)

order_sales["order_value"] = (
    order_sales["product_revenue"]
    + order_sales["freight_total"]
)

order_sales["order_month"] = (
    order_sales["order_purchase_timestamp"]
    .dt.to_period("M").astype(str)
)

order_sales["month_name"] = (
    order_sales["order_purchase_timestamp"].dt.strftime("%B")
)


# Add average review score per order.
order_sales = order_sales.merge(
    order_reviews,
    on="order_id",
    how="left"
)

order_sales.to_csv(
    OUTPUT_DIR / "order_level_analysis.csv",
    index=False
)


# ============================================================
# 13. KEY BUSINESS METRICS
# ============================================================

total_revenue = fact["price"].sum()

total_orders = order_sales["order_id"].nunique()

total_customers = valid_orders["customer_id"].nunique()

total_products_sold = len(fact)

total_quantity = len(fact)

overall_aov = order_sales["order_value"].mean()

average_review = order_sales["review_score"].mean()

total_freight = fact["freight_value"].sum()

low_review_percentage = (
    order_sales["review_score"].isin([1, 2]).sum()
    / order_sales["review_score"].notna().sum()
    * 100
    if order_sales["review_score"].notna().sum() > 0
    else np.nan
)


# ============================================================
# 14. BUSINESS QUESTION 1
# Which product category generates the highest revenue?
# ============================================================

category_sales = (
    fact.groupby("product_category_name_english", as_index=False)
    .agg(
        total_revenue=("price", "sum"),
        quantity_sold=("order_item_id", "count")
    )
    .sort_values("total_revenue", ascending=False)
)

top_category = category_sales.iloc[0]


# ============================================================
# 15. BUSINESS QUESTION 2
# Which month has the highest revenue?
# ============================================================

monthly_sales = (
    fact.groupby("order_month", as_index=False)
    .agg(
        monthly_revenue=("price", "sum"),
        orders=("order_id", "nunique")
    )
    .sort_values("order_month")
)

peak_month = monthly_sales.loc[
    monthly_sales["monthly_revenue"].idxmax()
]


# ============================================================
# 16. BUSINESS QUESTION 3
# Which state generates the highest revenue?
# ============================================================

regional_sales = (
    fact.groupby("customer_state", as_index=False)
    .agg(
        total_revenue=("price", "sum"),
        orders=("order_id", "nunique")
    )
    .sort_values("total_revenue", ascending=False)
)

top_state = regional_sales.iloc[0]

top_state_percentage = (
    top_state["total_revenue"] / total_revenue * 100
    if total_revenue > 0 else 0
)


# ============================================================
# 17. BUSINESS QUESTION 4
# What is the average order value?
# ============================================================

# AOV = Total order value / Number of orders
# Order value includes product price + freight.

aov = overall_aov


# ============================================================
# 18. BUSINESS QUESTION 5
# How satisfied are customers with their purchases?
# ============================================================

review_distribution = (
    order_sales["review_score"]
    .value_counts()
    .sort_index()
    .reindex([1, 2, 3, 4, 5], fill_value=0)
)

review_distribution_df = review_distribution.reset_index()
review_distribution_df.columns = ["review_score", "number_of_reviews"]

average_review_score = order_sales["review_score"].mean()

positive_reviews = (
    order_sales["review_score"].isin([4, 5]).sum()
)

total_reviewed_orders = order_sales["review_score"].notna().sum()

positive_review_percentage = (
    positive_reviews / total_reviewed_orders * 100
    if total_reviewed_orders > 0 else 0
)


# ============================================================
# 19. DISPLAY BUSINESS QUESTION ANSWERS
# ============================================================

answers = []

answers.append("OLIST E-COMMERCE SALES ANALYSIS")
answers.append(f"Analysis Period: {ANALYSIS_YEAR}")
answers.append("=" * 60)

answers.append("\nBUSINESS QUESTION 1")
answers.append(
    "Which product category generates the highest revenue?"
)
answers.append(
    f"Answer: {top_category['product_category_name_english']}"
)
answers.append(
    f"Revenue: R$ {top_category['total_revenue']:,.2f}"
)

answers.append("\nBUSINESS QUESTION 2")
answers.append(
    "Which month has the highest revenue?"
)
answers.append(
    f"Answer: {peak_month['order_month']}"
)
answers.append(
    f"Revenue: R$ {peak_month['monthly_revenue']:,.2f}"
)

answers.append("\nBUSINESS QUESTION 3")
answers.append(
    "Which state generates the highest revenue?"
)
answers.append(
    f"Answer: {top_state['customer_state']}"
)
answers.append(
    f"Revenue: R$ {top_state['total_revenue']:,.2f}"
)
answers.append(
    f"Share of total revenue: {top_state_percentage:.2f}%"
)

answers.append("\nBUSINESS QUESTION 4")
answers.append(
    "What is the average order value?"
)
answers.append(
    f"Answer: R$ {aov:,.2f}"
)

answers.append("\nBUSINESS QUESTION 5")
answers.append(
    "How satisfied are customers with their purchases?"
)
answers.append(
    f"Average review score: {average_review_score:.2f}/5"
)
answers.append(
    f"Positive reviews (4 or 5): {positive_review_percentage:.2f}%"
)
answers.append(
    f"Total reviewed orders: {total_reviewed_orders:,}"
)

answers.append("\n" + "=" * 60)
answers.append("ADDITIONAL BUSINESS METRICS")
answers.append(f"Total product revenue: R$ {total_revenue:,.2f}")
answers.append(f"Total orders: {total_orders:,}")
answers.append(f"Total customers: {total_customers:,}")
answers.append(f"Total items sold: {total_quantity:,}")
answers.append(f"Total freight: R$ {total_freight:,.2f}")
answers.append(
    f"Low reviews (1 or 2): {low_review_percentage:.2f}%"
)

answers_text = "\n".join(answers)

print("\n" + answers_text)

with open(
    OUTPUT_DIR / "business_questions_answers.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(answers_text)


# ============================================================
# 20. SAVE ADDITIONAL ANALYSIS CSV FILES
# ============================================================

category_sales.to_csv(
    OUTPUT_DIR / "category_sales.csv",
    index=False
)

monthly_sales.to_csv(
    OUTPUT_DIR / "monthly_sales.csv",
    index=False
)

regional_sales.to_csv(
    OUTPUT_DIR / "regional_sales.csv",
    index=False
)

review_distribution_df.to_csv(
    OUTPUT_DIR / "review_distribution.csv",
    index=False
)


# ============================================================
# 21. GENERATE KPI JSON FOR DASHBOARD
# ============================================================

kpis = {
    "analysis_year": ANALYSIS_YEAR,
    "total_revenue": round(float(total_revenue), 2),
    "total_orders": int(total_orders),
    "total_customers": int(total_customers),
    "total_items_sold": int(total_quantity),
    "average_order_value": round(float(overall_aov), 2),
    "average_review_score": (
        round(float(average_review), 2)
        if pd.notna(average_review) else None
    ),
    "total_freight": round(float(total_freight), 2),
    "positive_review_percentage": round(
        float(positive_review_percentage), 2
    ),
    "low_review_percentage": (
        round(float(low_review_percentage), 2)
        if pd.notna(low_review_percentage) else None
    )
}

with open(
    OUTPUT_DIR / "kpis.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(kpis, file, indent=4)


# ============================================================
# 22. CHART 1 - MONTHLY REVENUE
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=monthly_sales,
    x="order_month",
    y="monthly_revenue"
)

plt.title(
    f"Monthly Revenue - {ANALYSIS_YEAR}",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Revenue (R$)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    CHART_DIR / "monthly_revenue.png",
    dpi=300
)

plt.close()


# ============================================================
# 23. CHART 2 - TOP 10 PRODUCT CATEGORIES
# ============================================================

top_10_categories = category_sales.head(10)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_10_categories,
    x="total_revenue",
    y="product_category_name_english"
)

plt.title(
    "Top 10 Product Categories by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (R$)")
plt.ylabel("Product Category")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "top_categories.png",
    dpi=300
)

plt.close()


# ============================================================
# 24. CHART 3 - REVENUE BY STATE
# ============================================================

top_10_states = regional_sales.head(10)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_10_states,
    x="total_revenue",
    y="customer_state"
)

plt.title(
    "Top 10 States by Revenue",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Revenue (R$)")
plt.ylabel("Customer State")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "state_revenue.png",
    dpi=300
)

plt.close()


# ============================================================
# 25. CHART 4 - CUSTOMER REVIEW DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=review_distribution_df,
    x="review_score",
    y="number_of_reviews"
)

plt.title(
    "Customer Review Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "review_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 26. CHART 5 - MONTHLY ORDER COUNT
# ============================================================

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=monthly_sales,
    x="order_month",
    y="orders",
    marker="o"
)

plt.title(
    "Monthly Number of Orders",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    CHART_DIR / "monthly_orders.png",
    dpi=300
)

plt.close()


# ============================================================
# 27. CHART 6 - AVERAGE ORDER VALUE BY MONTH
# ============================================================

monthly_aov = (
    order_sales.groupby("order_month", as_index=False)
    .agg(
        average_order_value=("order_value", "mean")
    )
    .sort_values("order_month")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=monthly_aov,
    x="order_month",
    y="average_order_value",
    marker="o"
)

plt.title(
    "Average Order Value by Month",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Average Order Value (R$)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    CHART_DIR / "monthly_aov.png",
    dpi=300
)

plt.close()


# ============================================================
# 28. SAVE DATA CLEANING LOG
# ============================================================

cleaning_log_df = pd.DataFrame(cleaning_log)

cleaning_log_df.to_csv(
    OUTPUT_DIR / "data_cleaning_log.csv",
    index=False
)


# ============================================================
# 29. GENERATE BUSINESS INSIGHTS REPORT
# ============================================================

report = f"""
# Olist E-Commerce Sales Analysis

## Analysis Period
{ANALYSIS_YEAR}

## Key Performance Indicators

- Total Revenue: R$ {total_revenue:,.2f}
- Total Orders: {total_orders:,}
- Total Customers: {total_customers:,}
- Total Items Sold: {total_quantity:,}
- Average Order Value: R$ {overall_aov:,.2f}
- Average Review Score: {average_review_score:.2f}/5
- Total Freight: R$ {total_freight:,.2f}

## Business Question Findings

### 1. Highest Revenue Product Category
{top_category['product_category_name_english']}

Revenue: R$ {top_category['total_revenue']:,.2f}

### 2. Highest Revenue Month
{peak_month['order_month']}

Revenue: R$ {peak_month['monthly_revenue']:,.2f}

### 3. Highest Revenue State
{top_state['customer_state']}

Revenue: R$ {top_state['total_revenue']:,.2f}

Contribution to total revenue: {top_state_percentage:.2f}%

### 4. Average Order Value
R$ {overall_aov:,.2f}

### 5. Customer Satisfaction
Average review score: {average_review_score:.2f}/5

Positive reviews (4 or 5): {positive_review_percentage:.2f}%

## Files Generated

- cleaned_order_items.csv
- order_level_analysis.csv
- category_sales.csv
- monthly_sales.csv
- regional_sales.csv
- review_distribution.csv
- business_questions_answers.txt
- kpis.json
- data_cleaning_log.csv
- charts folder
"""

with open(
    OUTPUT_DIR / "business_insights_report.md",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"\nAnalysis year: {ANALYSIS_YEAR}")
print(f"Total revenue: R$ {total_revenue:,.2f}")
print(f"Total orders: {total_orders:,}")
print(f"Average order value: R$ {overall_aov:,.2f}")
print(f"Average review score: {average_review_score:.2f}/5")

print(f"\nAll output files are saved in:")
print(OUTPUT_DIR)

print("\nYour interactive dashboard can now use:")
print("- outputs/cleaned_order_items.csv")
print("- outputs/order_level_analysis.csv")
print("- outputs/kpis.json")

print("\nRun your dashboard separately using:")
print("streamlit run dashboard.py")