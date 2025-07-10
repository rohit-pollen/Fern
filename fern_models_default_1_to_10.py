import os
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.fern_static_variables import *
from utils.fern_preprocess import *
from utils.fern_train import train_sales_prob_price_model, train_domestic_export_model, plot_metrics_report, train_time_at_a_price_model,\
      train_amount_at_a_price_model
from utils.fern_misc import print_df_shapes_auto

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(BASE_DIR, 'secrets', 'default_key.json')

# credentials = service_account.Credentials.from_service_account_file(key_path)
# client = bigquery.Client(credentials=credentials)

def load_dict_from_json(filepath='default_key.json'):
    """
    Load a JSON file as a Python dictionary.
    Used for loading service account or credentials data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

loaded_key = load_dict_from_json('secrets/default_key.json')
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
creds = Credentials.from_authorized_user_info(loaded_key, scopes=SCOPES)
client = bigquery.Client(project="dev-sd-lake", credentials=creds)


def fetch_data():
    """
    Query BigQuery tables and return data as DataFrames.
    Includes product, seller, offer, and order information.
    """
    product_listings = client.query_and_wait(product_listings_query).to_dataframe()
    products = client.query_and_wait(products_query).to_dataframe()
    product_categories = client.query_and_wait(product_categories_query).to_dataframe()
    product_subcategories = client.query_and_wait(product_subcategories_query).to_dataframe()
    sellers = client.query_and_wait(sellers_query).to_dataframe()
    offers = client.query_and_wait(offers_query).to_dataframe()
    orders_level_1 = client.query_and_wait(orders_level_1_query).to_dataframe()
    orders_level_2 = client.query_and_wait(orders_level_2_query).to_dataframe()
    return product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2

def preprocess_data(product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2):
    """
    Perform cleaning, merging, and feature engineering on the raw data.
    Returns a processed and sampled DataFrame ready for model training.
    """
    sellers, offers, orders, product_categories, product_subcategories, product_listings = merge_data(product_listings, products,
                                                                                            product_categories, product_subcategories,sellers, offers,
                                                                                            orders_level_1, orders_level_2)
    df_inv, non_core_inv = split_data(product_listings)
    non_core_inv, offers = clean_non_core_data(non_core_inv, offers)
    df_inv, offers = clean_core_data(df_inv, offers)
    total_inv = merge_core_and_non_core_data(df_inv, non_core_inv)

    total_inv = get_target(total_inv)
    print_df_shapes_auto(total_inv)

    total_inv = clean_merged_total_inv_data(total_inv)
    total_inv = impute_domestic_export_feature(total_inv)

    inv_with_offer, inv_without_offer = get_inv_with_offers_and_without_offers(total_inv)

    inv_without_offer = get_price_sampling_inv_without_offer_plus_time_and_units_correction(inv_without_offer)
    inv_with_offer = get_price_sampling_inv_with_offer_plus_time_and_units_correction(inv_with_offer)
    inv_with_offer['offer_day_diff_updated_inv'] = inv_with_offer.apply(get_updated_time_for_offers, axis = 1)
    inv_with_offer['total_units'] = inv_with_offer.apply(get_updated_total_units_for_offers, axis = 1)
    inv_with_offer = update_sellability_for_offers(inv_with_offer)

    total_inv_after_sampled = merge_inv_with_and_without_offers_after_sampling(inv_without_offer, inv_with_offer)
    total_inv_after_sampled = get_listing_condition_feat(total_inv_after_sampled)

    return total_inv_after_sampled

def train(total_inv_after_sampled):
    """
    Train both the sales probability model and the domestic/export model.
    Also plots and saves metrics reports for each model.
    """
    sales_prob_price_model, sales_prob_y_val, sales_prob_pred_probs = train_sales_prob_price_model(total_inv_after_sampled)
    plot_metrics_report(sales_prob_y_val, sales_prob_pred_probs, t=0.65, model_name="sales_prob_price_model")

    domestic_export_price_model, domestic_export_y_val, domestic_export_pred_probs = train_domestic_export_model(total_inv_after_sampled)
    plot_metrics_report(domestic_export_y_val, domestic_export_pred_probs, t=0.65, model_name="domestic_export_price_model")

    time_at_a_price_model, time_at_a_price_inverse_y_val, time_preds = train_time_at_a_price_model(total_inv_after_sampled)
    amount_at_a_price_model, amount_at_a_price_inverse_y_val, amount_preds = train_amount_at_a_price_model(total_inv_after_sampled)

    return sales_prob_price_model, domestic_export_price_model, time_at_a_price_model, amount_at_a_price_model

def main():
    """
    Main function to execute the full pipeline: fetch → preprocess → train.
    Returns the trained model objects for downstream use.
    """
    product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2 = fetch_data()
    print_df_shapes_auto(product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2)
    total_inv_after_sampled = preprocess_data(product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2)
    sales_prob_price_model, domestic_export_price_model, time_at_a_price_model, amount_at_a_price_model = train(total_inv_after_sampled)
    return sales_prob_price_model, domestic_export_price_model, time_at_a_price_model, amount_at_a_price_model

if __name__ == "__main__":
    sales_prob_price_model, domestic_export_price_model, time_at_a_price_model, amount_at_a_price_model = main()
    date_stamp = datetime.datetime.now().strftime("%Y%m%d")

    sales_prob_price_model_filename = f"models/sales_prob_price_model_{date_stamp}.pkl"
    domestic_export_price_model_filename = f"models/domestic_export_price_model_{date_stamp}.pkl"
    time_at_a_price_model_filename = f"models/time_at_a_price_model_{date_stamp}.pkl"
    amount_at_a_price_model_filename = f"models/amount_at_a_price_model_{date_stamp}.pkl"

    joblib.dump(sales_prob_price_model, sales_prob_price_model_filename)
    joblib.dump(domestic_export_price_model, domestic_export_price_model_filename)
    joblib.dump(time_at_a_price_model, time_at_a_price_model_filename)
    joblib.dump(amount_at_a_price_model, amount_at_a_price_model_filename)