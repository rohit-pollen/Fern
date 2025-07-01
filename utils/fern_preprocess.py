import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from utils.fern_misc import print_df_shapes_auto
from utils.fern_static_variables import seller_name_to_short_form

# adding helper functions for default 1 to 6

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


def dropping_null_brand_cat_subcats(df_inv):
    df_inv = df_inv[~df_inv.brand.isnull()]
    df_inv = df_inv[~df_inv.product_category.isnull()]
    df_inv = df_inv[~df_inv.product_subcategory.isnull()]
    return df_inv

def cols_to_lower_rem_space(df, cols_list):
    for col in ['sku_name', 'product_name', 'brand', 'brand_', 'product_description', 'product_category', 'product_sub_category', 'product_subcategory',\
                'warehouse_location', 'buyer', 'seller', 'package_type', 'deal_type', 'origin_', 'destination', 'region_of_export', \
                'domestic_export', 'order_type', 'country', 'currency', 'country_of_origin', 'pack_label_language', 'dangerous_goods\n(y/n)', 'shelf_life_bucket',
                'relavant_(r)/ir-relavant(ir)', 'product_restricted_countries', 'store_name', 'quarter', 'temp_reference', 'tracking_no', 'offer_type', 'fiscal_month',
                'proposed_buyer_product_preferences', 'product_shelf_life_months', 'priority', 'seller', 'lms_seller_id', 'persona', 'core_vs._\nnon_core', 'sku_product_name',
                'warehouse_country', 'warehouse_address', 'measurement_units', 'listing_currency', 'manufacturing_country', 'lbh_measurement_units', 'package_type', 'seller_name',
                'pack_size_unit', 'package_labeled_language', 'inventory_class', 'description', 'unit_dimensions']:
                if col in cols_list:
                    df[col] = df[col].str.lower().str.replace(' ', '_')
    return df

def remove_starting_and_trailing_underscores(x):
    if x != '':
        if x[0] == '_':
            for i in range(len(x)):
                if x[i] != '_':
                    start = i
                    break
            x = x[start:]
        if x[-1] == '_':
            for i in reversed(range(len(x))):
                if x[i] != '_':
                    end = i
                    break
            x = x[:end+1]
    return x

def cleaning_underscores_inv_data(df_inv, col_list):
    for col in col_list:
        print(col)
        df_inv[col] = df_inv[col].apply(remove_starting_and_trailing_underscores)
    return df_inv

def drop_dups_and_clean(df_inv):
    print('Dropping duplicates ...')
    df_inv = df_inv.drop_duplicates()
    return df_inv

def split_data(product_listings):
    df_inv = product_listings[product_listings.inventory_class == 'CORE']
    non_core_inv = product_listings[product_listings.inventory_class == 'NON_CORE']
    df_inv = df_inv.reset_index(drop = True)
    non_core_inv = non_core_inv.reset_index(drop = True)
    return df_inv, non_core_inv


def merge_non_core_inv_and_offer(non_core_inv, df_offer):
    print(df_offer.columns)
    temp = pd.merge(non_core_inv, df_offer[['sku_number', 'seller_name', 'date_of_offer', 'actual', 'tracking_no', 'buyer', 'total_units_ordered',
                                            'recovery_rate_percentage', 'offer_price_per_unit_usd', 'total_order_price_usd', 'domestic_export']], 'left', on = ['sku_number', 'seller_name'])
    # temp['updated_on'] = pd.to_datetime(temp['updated_on'])
    temp['offer_day_diff_updated_inv'] = (temp['date_of_offer'] - temp['updated_on']).dt.days
    # avoiding future entry from same sku and seller to get filtered out hence nulling it as will remain in the dataframe
    temp['offer_day_diff_updated_inv'] = np.where(temp.offer_day_diff_updated_inv < 0, np.nan, temp['offer_day_diff_updated_inv'])
    temp = temp[(temp.offer_day_diff_updated_inv >= 0) | (temp.offer_day_diff_updated_inv.isnull())]
    df_non_core_inv_min_offer_day_diff_updated_inv = temp.groupby(['sku_number', 'seller_name', 'date_of_offer']).agg({'offer_day_diff_updated_inv' : 'min'}).reset_index()\
                                        .rename(columns = {'offer_day_diff_updated_inv' : 'min_offer_day_diff_updated_inv'})
    temp = pd.merge(temp, df_non_core_inv_min_offer_day_diff_updated_inv, 'left', on = ['sku_number', 'seller_name', 'date_of_offer'])
    temp = temp[(temp.offer_day_diff_updated_inv == temp.min_offer_day_diff_updated_inv) | (temp.offer_day_diff_updated_inv.isnull())]
    temp.drop('min_offer_day_diff_updated_inv', axis = 1, inplace = True)
    return temp

def remove_percent_sign(x):
    if pd.isna(x):
        return x
    else:
        return x[:-1]
  
def clean_non_core_data(non_core_inv, offers):

    # non core cleaning
    non_core_inv = cols_to_lower_rem_space(non_core_inv, non_core_inv.columns)
    non_core_inv = non_core_inv.rename(columns = {'total_number_of_items' : 'total_units'})
    non_core_inv['updated_on'] = pd.to_datetime(non_core_inv['updated_on'])
    non_core_inv['expiry_date'] = pd.to_datetime(non_core_inv['expiry_date'])

    non_core_inv = non_core_inv.drop_duplicates(subset = ['sku_number', 'expiry_date', 'updated_on', 'seller_name'])
    print_df_shapes_auto(non_core_inv)

    offers = offers.rename(columns = {'seller' : 'seller_name'})
    offers['date_of_offer'] = pd.to_datetime(offers['date_of_offer'])
    offers['seller_name'] = offers['seller_name'].str.lower()
    offers['total_units_ordered'] = offers['total_units_ordered'].str.replace(',', '')
    offers['total_units_ordered'] = offers['total_units_ordered'].astype(float)

    non_core_inv = merge_non_core_inv_and_offer(non_core_inv, offers)
    non_core_inv['total_units_ordered'] = non_core_inv['total_units_ordered'].fillna(0)
    non_core_inv['sell_thru_rate'] = non_core_inv['total_units_ordered'] / non_core_inv['total_units']
    non_core_inv['sell_thru_rate'] = np.where(non_core_inv['sell_thru_rate'] > 1, 1, non_core_inv['sell_thru_rate'])
    non_core_inv['shelf_life_remaining_days'] = (non_core_inv['expiry_date'] - non_core_inv['updated_on']).dt.days

    non_core_inv['recovery_rate_percentage'] = non_core_inv['recovery_rate_percentage'].apply(remove_percent_sign)
    non_core_inv['recovery_rate_percentage'] = np.where(non_core_inv['recovery_rate_percentage'] == '', np.nan, non_core_inv['recovery_rate_percentage'])
    non_core_inv['recovery_rate_percentage'] = non_core_inv['recovery_rate_percentage'].astype(float)
    non_core_inv['recovery_rate_percentage'] = np.where(non_core_inv['recovery_rate_percentage'].isnull(), 0 , non_core_inv['recovery_rate_percentage'])

    return non_core_inv, offers

def merge_inv_and_offer(df_inv, df_offer):
    df_inv = pd.merge(df_inv, df_offer[['sku_number', 'expiry_date', 'seller_short', 'date_of_offer', 'actual', 'tracking_no', 'buyer', 'total_units_ordered',
                                        'recovery_rate_percentage', 'offer_price_per_unit_usd', 'total_order_price_usd', 'domestic_export']], 'left', on = ['sku_number', 'expiry_date', 'seller_short'])
    df_inv['offer_day_diff_updated_inv'] = (df_inv['date_of_offer'] - df_inv['updated_on']).dt.days
    df_inv = df_inv[(df_inv.offer_day_diff_updated_inv >= 0) | (df_inv.offer_day_diff_updated_inv.isnull())]
    df_min_offer_day_diff_updated_inv = df_inv.groupby(['sku_number', 'expiry_date', 'seller_short', 'date_of_offer']).agg({'offer_day_diff_updated_inv' : 'min'}).reset_index()\
                                        .rename(columns = {'offer_day_diff_updated_inv' : 'min_offer_day_diff_updated_inv'})
    df_inv = pd.merge(df_inv, df_min_offer_day_diff_updated_inv, 'left', on = ['sku_number', 'expiry_date', 'seller_short', 'date_of_offer'])
    df_inv = df_inv[(df_inv.offer_day_diff_updated_inv == df_inv.min_offer_day_diff_updated_inv) | (df_inv.offer_day_diff_updated_inv.isnull())]
    df_inv.drop('min_offer_day_diff_updated_inv', axis = 1, inplace = True)
    return df_inv

def apply_seller_short(x):
    if x in seller_name_to_short_form.keys():
        return seller_name_to_short_form[x]
    else:
        return x
    
def clean_core_data(df_inv, offers):
    # core cleaning
    df_inv = dropping_null_brand_cat_subcats(df_inv)
    df_inv = cols_to_lower_rem_space(df_inv, df_inv.columns)
    df_inv = drop_dups_and_clean(df_inv)
    df_inv['updated_on'] = pd.to_datetime(df_inv['updated_on'])
    df_inv['seller_short'] = df_inv['seller_name'].apply(apply_seller_short)
    df_inv['expiry_date'] = pd.to_datetime(df_inv['expiry_date'])

    offers = offers.rename(columns = {'seller_name' : 'seller_short'})
    offers = offers[offers['expiry_date'] != '-']
    offers['expiry_date'] = pd.to_datetime(offers['expiry_date'], format = "%d-%b-%Y")

    df_inv = merge_inv_and_offer(df_inv, offers)
    df_inv['total_units_ordered'] = df_inv['total_units_ordered'].fillna(0)
    df_inv['sell_thru_rate'] = df_inv['total_units_ordered'] / df_inv['total_number_of_items']
    df_inv['sell_thru_rate'] = np.where(df_inv['sell_thru_rate'] > 1, 1, df_inv['sell_thru_rate'])
    df_inv['shelf_life_remaining_days'] = (df_inv['expiry_date'] - df_inv['updated_on']).dt.days

    df_inv['recovery_rate_percentage'] = df_inv['recovery_rate_percentage'].apply(remove_percent_sign)
    df_inv['recovery_rate_percentage'] = np.where(df_inv['recovery_rate_percentage'] == '', np.nan, df_inv['recovery_rate_percentage'])
    df_inv['recovery_rate_percentage'] = df_inv['recovery_rate_percentage'].astype(float)
    df_inv['recovery_rate_percentage'] = np.where(df_inv['recovery_rate_percentage'].isnull(), 0 , df_inv['recovery_rate_percentage'])

    # removing some rows which are common on ['sku_number', 'expiry_date', 'seller_short', 'updated_on']. Why ?
    # when matching inventory to offers, the inventory which is offered multiple times gets repeated. Which is fine and we wish to consider that
    # but there are still some combos in inv which were not offered, buyer is null for them, still they are repeated with almost the same complete row. We need to remove such rows

    temp = df_inv[df_inv.duplicated(subset = ['sku_number', 'expiry_date', 'seller_short', 'updated_on'])]
    to_remove_ids = temp[temp.buyer.isnull()].id
    print(len(to_remove_ids))
    df_inv = df_inv[~df_inv.id.isin(to_remove_ids)]

    df_inv = df_inv.rename(columns = {'total_number_of_items' : 'total_units'})
    df_inv['updated_on'] = pd.to_datetime(df_inv['updated_on'])
    df_inv['expiry_date'] = pd.to_datetime(df_inv['expiry_date'])
    df_inv = df_inv[df_inv.sku_number != '']
    df_inv['seller_name'] = df_inv['seller_short']
    df_inv = df_inv.drop('seller_short', axis = 1)

    print_df_shapes_auto(df_inv)

    return df_inv, offers

def merge_core_and_non_core_data(df_inv, non_core_inv):
    total_inv = pd.concat([df_inv, non_core_inv], ignore_index=True)
    return total_inv

def get_target(df):
    df['sellability'] = np.where(df.date_of_offer.isnull(), 0, 1)
    print(df.sellability.value_counts())
    return df

def get_order_price_per_unit(df):
    if df['total_units_ordered'] == 0:
        return 0
    else:
        return df['total_order_price_usd'] / df['total_units_ordered']
    
def get_clusters(total_inv):
    # Step 2: Columns to exclude from clustering
    cols_to_ignore = [
        'id', 'product_id', 'seller_id', 'record_id', 'expiry_date',
        'package_labeled_language', 'product_restriction', 'updated_on',
        'date_of_offer', 'actual', 'tracking_no', 'buyer']

    # Step 3: Create clustering dataset
    df_cluster = total_inv.drop(columns=cols_to_ignore)

    # Step 4: Encode categorical columns (excluding domestic_export)
    categorical_cols = df_cluster.select_dtypes(include='object').columns.tolist()
    if 'domestic_export' in categorical_cols:
        categorical_cols.remove('domestic_export')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_cluster[col] = df_cluster[col].astype(str)
        df_cluster[col] = le.fit_transform(df_cluster[col])
        label_encoders[col] = le

    # Step 5: Fill missing values for clustering
    df_cluster.fillna(0, inplace=True)

    # Step 6: Scale features (excluding domestic_export)
    X_features = df_cluster.drop(columns=['domestic_export'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Step 7: Run KMeans with fixed K = 10
    k = 100
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    total_inv['cluster'] = kmeans.fit_predict(X_scaled)
    print(f"\n KMeans clustering completed with K = {k}")
    print('score - ', silhouette_score(X_scaled, total_inv['cluster']))
    return total_inv

def clean_merged_total_inv_data(total_inv):
    total_inv['total_order_price_usd'] = total_inv['total_order_price_usd'].str.replace(',', '')
    total_inv['total_order_price_usd'] = total_inv['total_order_price_usd'].astype(float)

    total_inv['order_price_per_unit_usd'] = total_inv.apply(get_order_price_per_unit, axis = 1)

    total_inv['offer_price_per_unit_usd'] = total_inv['offer_price_per_unit_usd'].str.replace(',', '')
    total_inv['offer_price_per_unit_usd'] = total_inv['offer_price_per_unit_usd'].astype(float)
    total_inv = total_inv[~total_inv['retail_price_per_unit_usd'].isin([0, np.nan])]

    total_inv = get_clusters(total_inv)

    return total_inv

def impute_domestic_export(row, df):
    if pd.notna(row['domestic_export']):
        return row['domestic_export']
    cluster_rows = df[(df['cluster'] == row['cluster']) & (df['domestic_export'].notna())]
    if not cluster_rows.empty:
        return cluster_rows['domestic_export'].mode()[0]
    return 'Domestic'

def impute_domestic_export_feature(total_inv):
    total_inv['domestic_export'] = total_inv.apply(lambda row: impute_domestic_export(row, total_inv), axis=1)
    total_inv['domestic_export_imputed'] = total_inv['domestic_export'].isna() & total_inv['domestic_export'].notna()
    total_inv.drop(columns=['cluster'], inplace=True)
    total_inv['domestic_export'] = total_inv['domestic_export'].str.lower()
    return total_inv

def get_inv_with_offers_and_without_offers(total_inv):
    inv_without_offer = total_inv[total_inv.sellability == 0]
    inv_with_offer = total_inv[total_inv.sellability == 1]
    return inv_with_offer, inv_without_offer


def get_dummy_prices_for_inv_without_offers(df):
    return list(np.random.uniform(low = df['asking_price_per_unit_usd'], high = df['retail_price_per_unit_usd'], size = (2,))) + [df['asking_price_per_unit_usd']]

def get_price_sampling_inv_without_offer(inv_without_offer):
    inv_without_offer['test_price_per_unit_usd'] = inv_without_offer.apply(get_dummy_prices_for_inv_without_offers, axis = 1)

    inv_without_offer = inv_without_offer.explode('test_price_per_unit_usd')
    inv_without_offer['sellability'] = np.where((inv_without_offer['test_price_per_unit_usd'] == inv_without_offer['asking_price_per_unit_usd']), 1, inv_without_offer['sellability'])
    return inv_without_offer

def get_dummy_prices_for_inv_with_offers(df):
    if df['total_units_ordered'] != 0:
        return list(np.random.uniform(low = df['order_price_per_unit_usd'], high = df['asking_price_per_unit_usd'], size = (2,))) + [df['order_price_per_unit_usd']] + [df['asking_price_per_unit_usd']] +\
            list(np.random.uniform(low = df['asking_price_per_unit_usd'], high = df['retail_price_per_unit_usd'], size = (2,)))
    else:
        return list(np.random.uniform(low = 0, high = df['offer_price_per_unit_usd'], size = (2,))) + [df['offer_price_per_unit_usd']] + [df['asking_price_per_unit_usd']] +\
            list(np.random.uniform(low = df['asking_price_per_unit_usd'], high = df['retail_price_per_unit_usd'], size = (2,)))

def get_price_sampling_inv_with_offer(inv_with_offer):
    inv_with_offer = inv_with_offer[~inv_with_offer['offer_price_per_unit_usd'].isin([0, np.nan])]
    inv_with_offer['test_price_per_unit_usd'] = inv_with_offer.apply(get_dummy_prices_for_inv_with_offers, axis = 1)

    inv_with_offer = inv_with_offer.explode('test_price_per_unit_usd')

    # for inv_with_offer where units have been ordered (converted offers), not just order_price_per_unit_usd is sellable but prices above that also sellable
    # but where units have NOT been ordered (non-converted offers), not just the offer_price_per_unit_usd is not sellable but also the prices less than offer_price_per_unit_usd are also not sellable

    inv_with_offer['sellability'] = np.where((inv_with_offer['total_units_ordered'] == 0) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['offer_price_per_unit_usd']),
                                            0, inv_with_offer['sellability'])

    inv_with_offer['sellability'] = np.where( ((inv_with_offer['total_units_ordered'] == 0) &
    ((inv_with_offer['test_price_per_unit_usd'] > inv_with_offer['asking_price_per_unit_usd']) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['retail_price_per_unit_usd'])) ),
                                            0, inv_with_offer['sellability'])

    inv_with_offer['sellability'] = np.where( ( (inv_with_offer['total_units_ordered'] != 0) &
    ((inv_with_offer['test_price_per_unit_usd'] > inv_with_offer['asking_price_per_unit_usd']) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['retail_price_per_unit_usd'])) ),
                                            0, inv_with_offer['sellability'])
    return inv_with_offer


def merge_data_after_sample(inv_without_offer, inv_with_offer):
    total_inv_after_sampled = pd.concat([inv_without_offer, inv_with_offer], ignore_index=True)
    total_inv_after_sampled['order_price_per_unit_usd'] = total_inv_after_sampled['test_price_per_unit_usd']
    total_inv_after_sampled = total_inv_after_sampled.drop('test_price_per_unit_usd', axis = 1)
    total_inv_after_sampled = total_inv_after_sampled.rename(columns = {'offer_day_diff_updated_inv' : 'time'})
    return total_inv_after_sampled


def get_listing_type(df):
    if df['shelf_life_remaining_days'] > 365*2:
        return 'fresh'
    elif df['shelf_life_remaining_days'] > 365 and df['recovery_rate_percentage'] > 0 and df['recovery_rate_percentage'] < 10:
        return 'obsolete'
    else:
        return 'excess'

def get_listing_condition_feat(total_inv_after_sampled):
    total_inv_after_sampled['listing_condition'] = total_inv_after_sampled.apply(get_listing_type, axis = 1)

    number_of_rows_to_randomize = int(total_inv_after_sampled.shape[0]*0.05)

    change = total_inv_after_sampled.sample(number_of_rows_to_randomize//2).index
    total_inv_after_sampled.loc[change,'listing_condition'] = 'damaged'
    change = total_inv_after_sampled.sample(number_of_rows_to_randomize//2).index
    total_inv_after_sampled.loc[change,'listing_condition'] = 'made_to_order'

    print(total_inv_after_sampled['listing_condition'].value_counts())

    return total_inv_after_sampled


# adding helper functions for default 7 to 10

def get_price_sampling_inv_without_offer_plus_time_and_units_correction(inv_without_offer):
    inv_without_offer['test_price_per_unit_usd'] = inv_without_offer.apply(get_dummy_prices_for_inv_without_offers, axis = 1)
    inv_without_offer = inv_without_offer.explode('test_price_per_unit_usd')
    inv_without_offer['test_price_per_unit_usd'] = inv_without_offer['test_price_per_unit_usd'].astype(float)

    inv_without_offer['offer_day_diff_updated_inv'] = np.random.normal(loc=40, scale=3, size=inv_without_offer.shape[0])
    inv_without_offer = inv_without_offer[inv_without_offer['asking_price_per_unit_usd'] != 0]
    inv_without_offer['offer_day_diff_updated_inv'] = inv_without_offer['test_price_per_unit_usd'] / inv_without_offer['asking_price_per_unit_usd'] * inv_without_offer['offer_day_diff_updated_inv']
    inv_without_offer['total_units'] = inv_without_offer['asking_price_per_unit_usd'] / inv_without_offer['test_price_per_unit_usd'] * inv_without_offer['total_units']
    inv_without_offer['sellability'] = np.where((inv_without_offer['test_price_per_unit_usd'] == inv_without_offer['asking_price_per_unit_usd']), 1, inv_without_offer['sellability'])
    return inv_without_offer


def get_price_sampling_inv_with_offer_plus_time_and_units_correction(inv_with_offer):
    inv_with_offer = inv_with_offer[~inv_with_offer['offer_price_per_unit_usd'].isin([0, np.nan])]
    inv_with_offer['test_price_per_unit_usd'] = inv_with_offer.apply(get_dummy_prices_for_inv_with_offers, axis = 1)
    inv_with_offer = inv_with_offer.explode('test_price_per_unit_usd')
    inv_with_offer['test_price_per_unit_usd'] = inv_with_offer['test_price_per_unit_usd'].astype(float)
    inv_with_offer = inv_with_offer[inv_with_offer['asking_price_per_unit_usd'] != 0]
    return inv_with_offer

def get_updated_time_for_offers(df):
  if df['total_units_ordered'] != 0:
    return df['test_price_per_unit_usd'] / df['order_price_per_unit_usd'] * df['offer_day_diff_updated_inv']
  else:
    return df['test_price_per_unit_usd'] / df['asking_price_per_unit_usd'] * df['offer_day_diff_updated_inv']
  
def get_updated_total_units_for_offers(df):
    if df['total_units_ordered'] != 0:
        return df['order_price_per_unit_usd'] / df['test_price_per_unit_usd'] * df['total_units']
    else:
        return df['asking_price_per_unit_usd'] / df['test_price_per_unit_usd'] * df['total_units']


def update_sellability_for_offers(inv_with_offer):
    # for inv_with_offer where units have been ordered (converted offers), not just order_price_per_unit_usd is sellable but prices above that also sellable
    # but where units have NOT been ordered (non-converted offers), not just the offer_price_per_unit_usd is not sellable but also the prices less than offer_price_per_unit_usd are also not sellable

    inv_with_offer['sellability'] = np.where((inv_with_offer['total_units_ordered'] == 0) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['offer_price_per_unit_usd']),
                                            0, inv_with_offer['sellability'])

    inv_with_offer['sellability'] = np.where( ((inv_with_offer['total_units_ordered'] == 0) &
    ((inv_with_offer['test_price_per_unit_usd'] > inv_with_offer['asking_price_per_unit_usd']) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['retail_price_per_unit_usd'])) ),
                                            0, inv_with_offer['sellability'])

    inv_with_offer['sellability'] = np.where( ( (inv_with_offer['total_units_ordered'] != 0) &
    ((inv_with_offer['test_price_per_unit_usd'] > inv_with_offer['asking_price_per_unit_usd']) & (inv_with_offer['test_price_per_unit_usd'] <= inv_with_offer['retail_price_per_unit_usd'])) ),
                                            0, inv_with_offer['sellability'])
    
    return inv_with_offer


def merge_inv_with_and_without_offers_after_sampling(inv_without_offer, inv_with_offer):
    total_inv_after_sampled = pd.concat([inv_without_offer, inv_with_offer], ignore_index=True)
    total_inv_after_sampled['order_price_per_unit_usd'] = total_inv_after_sampled['test_price_per_unit_usd']
    total_inv_after_sampled = total_inv_after_sampled.drop('test_price_per_unit_usd', axis = 1)
    total_inv_after_sampled['order_price_per_unit_usd'] = total_inv_after_sampled['order_price_per_unit_usd'].astype('float')
    total_inv_after_sampled = total_inv_after_sampled.rename(columns = {'offer_day_diff_updated_inv' : 'time'})
    total_inv_after_sampled['time'] = total_inv_after_sampled['time'].astype('float')
    return total_inv_after_sampled