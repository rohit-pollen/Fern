product_listings_query = """
SELECT * FROM pollen.product_listings;
"""
products_query = """
SELECT * FROM pollen.products;
"""
product_categories_query = """
SELECT * FROM pollen.product_categories;
"""
product_subcategories_query = """
SELECT * FROM pollen.product_subcategories;
"""
sellers_query = """
SELECT * FROM pollen.sellers;
"""
offers_query = """
SELECT * FROM pollen.offers;
"""
orders_level_1_query = """
SELECT * FROM pollen.orders_level_1;
"""
orders_level_2_query = """
SELECT * FROM pollen.orders_level_2;
"""


cols_list_underscore_cleaning = ['sku_product_name', 'brand', 'product_category', 'product_subcategory', 'warehouse_country',
            'country_of_origin', 'pack_label_language', 'product_restricted_countries', 'shelf_life_bucket']

inv_cols_list_to_change_dtypes = ['qty_of_cartons', 'units_per_cartons', 'retail_price_per_case_(local)', 'retail_price_per_case_(usd)', 'asking_price_per_case_(local)',
                              'asking_price_per_case_(usd)', 'total_retail_price_(local)', 'total_retail_price_(usd)', 'total_asking_price_(local)',
                              'total_asking_price_(usd)', 'pack_size_(number)', 'total_cbm', 'cbm_per_case', 'package_dimensions_per_package_type_(length)',
                              'package_dimensions_per_case_(width)', 'package_dimensions_per_case_(height)', 'net_weight_\nper_unit(kg)', 'net_weight_per_case_(kg)',
                              'gross_weight_per_case_(kg)', 'total_net_weight_(kg)', 'total_gross_weight_(kg)', 'cases_per_pallet', 'number_of_pallets', 'discount',
                              'shelf_remaining_days']

record_attributes = ['sku_number', 'sku_name', 'brand', 'product_category', 'product_subcategory', 'retail_price_per_unit_local', 'currency']

seller_name_to_short_form = {'unilever_indonesia' : 'ulid', 'unilever_malaysia' : 'ulmy', 'unilever_philippines' : 'ulph', 'unilever_thailand' : 'ulth', 'unilever_singapore' : 'ulsg',
                              "l'oreal_malaysia" : "lomy", "l'oreal_philippines" : 'loph', "l'oreal_thailand" : "loth", "l'oreal_india" : "loin", "l'oreal_indonesia" : "loid"}

sales_prob_price_model_params = {
    'loss_function': 'Logloss',
    'learning_rate': 0.3,
    'colsample_bylevel': 0.6,
    'depth': 4,
    'min_data_in_leaf': 20,
    'subsample': 0.6,
    'od_wait': 30,
    'custom_metric': 'AUC:hints=skip_train~false',
    'random_seed': 42
}

domestic_export_price_model_params = {
    'loss_function': 'Logloss',
    'learning_rate': 0.3,
    'colsample_bylevel': 0.6,
    'depth': 4,
    'min_data_in_leaf': 20,
    'subsample': 0.6,
    'od_wait': 30,
    'custom_metric': 'AUC:hints=skip_train~false',
    'random_seed': 42
}


sales_prob_train_cols = ['sku_number', 'brand', 'product_category', 'product_subcategory', 'seller_name', 'total_units', 'shelf_life_remaining_days',
                                'time', 'listing_condition', 'retail_price_per_unit_usd', 'order_price_per_unit_usd']
sales_prob_cat_cols = ['sku_number', 'brand', 'product_category', 'product_subcategory', 'seller_name', 'listing_condition']
sales_prob_target_col = 'sellability'

domestic_export_train_cols = ['sku_number', 'brand', 'product_category', 'product_subcategory', 'seller_name', 'total_units', 'shelf_life_remaining_days',
              'time', 'listing_condition', 'retail_price_per_unit_usd', 'order_price_per_unit_usd', 'domestic_export']
domestic_export_cat_cols = ['sku_number', 'brand', 'product_category', 'product_subcategory', 'seller_name', 'listing_condition', 'domestic_export']
domestic_export_target_col = 'sellability'