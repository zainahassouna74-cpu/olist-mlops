def build_ml_table(tables):
    orders = tables["olist_orders_dataset"]
    customers = tables["olist_customers_dataset"]
    items = tables["olist_order_items_dataset"]
    payments = tables["olist_order_payments_dataset"]
    reviews = tables["olist_order_reviews_dataset"]
    products = tables["olist_products_dataset"]
    sellers = tables["olist_sellers_dataset"]
    translation = tables["product_category_name_translation"]

    # Orders + customers
    joined_df = orders.merge(
        customers,
        on="customer_id",
        how="left",
    )

    # Aggregate order items
    items_agg = (
        items.groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )

    joined_df = joined_df.merge(
        items_agg,
        on="order_id",
        how="left",
    )

    # Aggregate payments
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_count=("payment_sequential", "count"),
            total_payment=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
        )
        .reset_index()
    )

    joined_df = joined_df.merge(
        payments_agg,
        on="order_id",
        how="left",
    )

    # Aggregate reviews
    reviews_agg = (
        reviews.groupby("order_id")
        .agg(
            review_count=("review_id", "count"),
            avg_review_score=("review_score", "mean"),
        )
        .reset_index()
    )

    joined_df = joined_df.merge(
        reviews_agg,
        on="order_id",
        how="left",
    )

    # Products + translated categories
    products_info = products.merge(
        translation,
        on="product_category_name",
        how="left",
    )

    # Product and seller information per item
    items_details = items.merge(
        products_info,
        on="product_id",
        how="left",
    ).merge(
        sellers,
        on="seller_id",
        how="left",
    )

    product_seller_agg = (
        items_details.groupby("order_id")
        .agg(
            unique_products=("product_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
            unique_categories=("product_category_name", "nunique"),
        )
        .reset_index()
    )

    joined_df = joined_df.merge(
        product_seller_agg,
        on="order_id",
        how="left",
    )

    return joined_df
