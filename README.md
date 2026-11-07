# Olist E-Commerce Sales Analysis — Project 01

## 1. Project Overview

This project analyzes the Olist Brazilian E-Commerce dataset using Python to understand sales performance, product categories, regional sales, order value, and customer reviews.

An interactive dashboard is developed using **Streamlit** to present the analysis and business insights.

**Analysis Period:** January 2017 – December 2017

## 2. Dataset

Download the dataset from [Kaggle – Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Place these six CSV files inside `data/`:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`
- `product_category_name_translation.csv`

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Plotly
- Excel
- Streamlit

## 4. Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## 5. Run the Project

Run the analysis:

```bash
python analysis.py
```

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## 6. Project Outputs

The `outputs/` folder contains:

- `dashboard.html` – Interactive dashboard
- `business_insights_report.md` – Business insights and recommendations
- `kpi_summary.json` – Key performance indicators
- `cleaning_log.json` – Data-cleaning details
- CSV files – Analysis results
- `charts/` – Generated visualizations

The Streamlit dashboard displays key metrics, filters, and interactive charts for exploring the sales data.

## 7. Data Cleaning

- Removed duplicate records.
- Handled missing and invalid values.
- Converted date columns.
- Handled missing prices and freight values.
- Translated product categories into English.
- Aggregated reviews at the order level.
- Excluded canceled and unavailable orders.

## 8. Business Questions

1. How does monthly revenue change?
2. Which product categories generate the highest revenue?
3. Which customer states generate the most revenue?
4. What is the average order value (AOV)?
5. What does the customer review distribution show?

## 9. Metric Definitions

- **Product Revenue:** Sum of product prices, excluding freight.
- **AOV:** Average order value, including product prices and freight.
- **Best Region:** Customer state with the highest product revenue.
- **Review Distribution:** One average review score per order.