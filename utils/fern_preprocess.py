import pandas as pd
import numpy as np
from utils.fern_misc import print_df_shapes_auto

def get_order_date(df_order_sheet1, df_order_sheet2):
  order_date_to_tracking_no_mapping = df_order_sheet1[['tracking_no', 'order_date']].drop_duplicates()
  df_order_sheet2 = pd.merge(df_order_sheet2, order_date_to_tracking_no_mapping, 'left',  on = ['tracking_no'])
  df_order_sheet2.drop('date_of_order', axis = 1, inplace = True)
  df_order = df_order_sheet2.copy()
  return df_order

def merge_data(product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2):
    sellers = sellers[['seller_name', 'persona_seller_type']].drop_duplicates()
    orders = get_order_date(orders_level_1, orders_level_2)
    orders = orders.drop_duplicates(subset = ['sku_number', 'expiry_date', 'tracking_no'])
    offers = offers.drop_duplicates(subset = ['sku_number', 'expiry_date', 'tracking_no'])
    offers = offers.rename(columns = {'total_units' : 'total_units_offered'})
    orders = orders.rename(columns = {'total_units' : 'total_units_ordered', 'total_offer_price_usd' : 'total_order_price_usd'})

    offers = pd.merge(offers, orders[['sku_number', 'expiry_date', 'tracking_no', 'total_units_ordered', 'recovery_rate_percentage',
                                    'total_order_price_usd', 'domestic_export']], 'left', on = ['sku_number', 'expiry_date', 'tracking_no'])

    product_subcategories = product_subcategories.rename(columns = {'id' : 'subcategory_id', 'name' : 'product_subcategory'})
    product_categories = product_categories.rename(columns = {'id' : 'category_id', 'name' : 'product_category'})

    products = pd.merge(products, product_categories, 'left', on = 'category_id')
    products = pd.merge(products, product_subcategories[['subcategory_id', 'product_subcategory']], 'left', on = 'subcategory_id')
    products = products.rename(columns = {'id' : 'product_id'})

    products = products.drop(['parent_product_id', 'created_at', 'updated_at', 'category_id', 'subcategory_id'], axis = 1)
    product_listings = product_listings.drop(['created_at', 'updated_at', 'pollen_updated_price_per_unit_local',
                        'barcode', 'barcode_key', 'manufactured_date', 'batch_number', 'image_links', 'usd_conversion', 'scoring'], axis = 1)
    product_listings = pd.merge(product_listings, products, 'left', on = 'product_id')

    print_df_shapes_auto(sellers, offers, orders, product_categories, product_subcategories, product_listings)
    print('----------------------------------')
    print(product_listings.inventory_class.value_counts())

    return sellers, offers, orders, product_categories, product_subcategories, product_listings


