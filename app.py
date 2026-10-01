import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Synthetic Transaction Mining App",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛒 Synthetic Transaction Mining App")

st.write(
    "Generate synthetic transaction data and discover "
    "product purchasing patterns using Association Rule Mining."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Settings")

number_of_transactions = st.sidebar.slider(
    "Number of Transactions",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

minimum_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01
)

minimum_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05
)


# ============================================================
# PRODUCT LIST
# ============================================================

products = [
    "Milk",
    "Bread",
    "Eggs",
    "Butter",
    "Cheese",
    "Rice",
    "Biscuits",
    "Coffee",
    "Tea",
    "Sugar",
    "Soap",
    "Shampoo",
    "Toothpaste",
    "Juice",
    "Chips"
]


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

def generate_data(number_of_transactions):

    transactions = []

    start_date = datetime(2025, 1, 1)

    for transaction_id in range(
        1,
        number_of_transactions + 1
    ):

        date = start_date + timedelta(
            days=random.randint(0, 180)
        )

        customer_id = random.randint(
            1,
            max(100, number_of_transactions // 3)
        )

        number_of_items = random.randint(2, 6)

        selected_products = random.sample(
            products,
            number_of_items
        )

        for product in selected_products:

            quantity = random.randint(1, 3)

            price = round(
                random.uniform(10, 500),
                2
            )

            transactions.append({
                "TransactionID": transaction_id,
                "CustomerID": customer_id,
                "Date": date.strftime("%Y-%m-%d"),
                "Product": product,
                "Quantity": quantity,
                "Price": price,
                "TotalAmount": round(
                    quantity * price,
                    2
                )
            })

    return pd.DataFrame(transactions)


# ============================================================
# GENERATE DATA BUTTON
# ============================================================

if st.button(
    "🚀 Generate Transaction Data",
    use_container_width=True
):

    df = generate_data(
        number_of_transactions
    )

    st.session_state["data"] = df

    st.success(
        "Synthetic transaction data generated successfully!"
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if "data" in st.session_state:

    df = st.session_state["data"]


    # ========================================================
    # DATASET
    # ========================================================

    st.header("📊 Transaction Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        df["TransactionID"].nunique()
    )

    col2.metric(
        "Customers",
        df["CustomerID"].nunique()
    )

    col3.metric(
        "Products",
        df["Product"].nunique()
    )

    col4.metric(
        "Revenue",
        f"₹{df['TotalAmount'].sum():,.2f}"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


    # ========================================================
    # DATA PREPROCESSING
    # ========================================================

    st.header("🧹 Data Preprocessing")

    cleaned_df = df.copy()

    original_records = len(cleaned_df)

    # Remove duplicate records
    cleaned_df = cleaned_df.drop_duplicates()

    # Remove missing values
    cleaned_df = cleaned_df.dropna()

    # Keep positive quantities
    cleaned_df = cleaned_df[
        cleaned_df["Quantity"] > 0
    ]

    # Keep positive prices
    cleaned_df = cleaned_df[
        cleaned_df["Price"] > 0
    ]

    cleaned_records = len(cleaned_df)

    col1, col2 = st.columns(2)

    col1.metric(
        "Original Records",
        original_records
    )

    col2.metric(
        "Cleaned Records",
        cleaned_records
    )

    st.success(
        "Data preprocessing completed successfully."
    )


    # ========================================================
    # PRODUCT SALES ANALYSIS
    # ========================================================

    st.header("📦 Product Sales Analysis")

    product_sales = (
        cleaned_df
        .groupby("Product")["Quantity"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    product_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Product Quantity Sold"
    )

    ax.set_xlabel("Product")

    ax.set_ylabel(
        "Quantity Sold"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # TRANSACTION BASKET
    # ========================================================

    st.header("🛍️ Transaction Basket")

    basket = (
        cleaned_df
        .groupby(
            [
                "TransactionID",
                "Product"
            ]
        )["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert quantity into 0/1
    basket = basket.map(
        lambda x: 1 if x > 0 else 0
    )

    st.write(
        "The basket represents whether a product "
        "was purchased in each transaction."
    )

    st.dataframe(
        basket.head(20),
        use_container_width=True
    )


    # ========================================================
    # ASSOCIATION RULE MINING
    # ========================================================

    st.header("🔍 Association Rule Mining")

    try:

        # -----------------------------------------------
        # APRIORI
        # -----------------------------------------------

        frequent_itemsets = apriori(
            basket,
            min_support=minimum_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:

            st.warning(
                "No frequent itemsets found. "
                "Try reducing Minimum Support."
            )

        else:

            st.subheader(
                "Frequent Itemsets"
            )

            # Create display copy
            frequent_display = (
                frequent_itemsets.copy()
            )

            frequent_display[
                "itemsets"
            ] = frequent_display[
                "itemsets"
            ].apply(
                lambda x:
                    ", ".join(sorted(x))
            )

            st.dataframe(
                frequent_display,
                use_container_width=True
            )


            # -----------------------------------------------
            # ASSOCIATION RULES
            # -----------------------------------------------

            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=minimum_confidence
            )


            if rules.empty:

                st.warning(
                    "No association rules found. "
                    "Try reducing Minimum Confidence."
                )

            else:

                # Create display version
                rules_display = rules.copy()

                rules_display[
                    "antecedents"
                ] = rules_display[
                    "antecedents"
                ].apply(
                    lambda x:
                        ", ".join(sorted(x))
                )

                rules_display[
                    "consequents"
                ] = rules_display[
                    "consequents"
                ].apply(
                    lambda x:
                        ", ".join(sorted(x))
                )


                # -------------------------------------------
                # ASSOCIATION RULE TABLE
                # -------------------------------------------

                st.subheader(
                    "Association Rules"
                )

                display_columns = [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]

                st.dataframe(
                    rules_display[
                        display_columns
                    ],
                    use_container_width=True
                )


                # -------------------------------------------
                # TOP RULES
                # -------------------------------------------

                st.subheader(
                    "⭐ Top Association Rules"
                )

                top_rules = (
                    rules_display
                    .sort_values(
                        by="lift",
                        ascending=False
                    )
                    .head(10)
                )

                st.dataframe(
                    top_rules[
                        display_columns
                    ],
                    use_container_width=True
                )


                # -------------------------------------------
                # DOWNLOAD RULES
                # -------------------------------------------

                csv = rules_display[
                    display_columns
                ].to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download Association Rules",
                    data=csv,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )


                # -------------------------------------------
                # BEST RULE
                # -------------------------------------------

                best_rule = rules_display.iloc[0]

                st.info(
                    f"Highest-lift rule: "
                    f"{best_rule['antecedents']} → "
                    f"{best_rule['consequents']} "
                    f"(Lift: {best_rule['lift']:.2f})"
                )


    except Exception as e:

        st.error(
            f"Mining error: {e}"
        )


# ============================================================
# NO DATA MESSAGE
# ============================================================

else:

    st.info(
        "👈 Use the sidebar settings and click "
        "'Generate Transaction Data' to start."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "Built using Python, Streamlit, Pandas, "
    "MLxtend and Association Rule Mining."
)
