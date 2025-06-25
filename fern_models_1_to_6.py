import os
from google.cloud import bigquery
from google.oauth2 import service_account
from utils.fern_static_variables import *
from utils.fern_misc import print_df_shapes_auto

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to sandbox_key.json inside 'secrets' directory
key_path = os.path.join(BASE_DIR, 'secrets', 'sandbox_key.json')
credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(credentials=credentials)



def fetch_data():
  product_listings = client.query_and_wait(product_listings_query).to_dataframe()
  products = client.query_and_wait(products_query).to_dataframe()
  product_categories = client.query_and_wait(product_categories_query).to_dataframe()
  product_subcategories = client.query_and_wait(product_subcategories_query).to_dataframe()
  sellers = client.query_and_wait(sellers_query).to_dataframe()
  offers = client.query_and_wait(offers_query).to_dataframe()
  orders_level_1 = client.query_and_wait(orders_level_1_query).to_dataframe()
  orders_level_2 = client.query_and_wait(orders_level_2_query).to_dataframe()
  return product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2

def preprocess_data():
   return

def train():
   return

def main():
   product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2 = fetch_data()
   print_df_shapes_auto(product_listings, products, product_categories, product_subcategories, sellers, offers, orders_level_1, orders_level_2)

   preprocess_data()

   train()
   
   return

if __name__ == "__main__":
   main()