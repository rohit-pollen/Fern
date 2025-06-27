from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import pandas as pd   # only needed for the helper at the end

# --- 1▸ paste your JSON here ---------------------------------------------
adc_info = {
    "type": "authorized_user",
    "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
    "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
    "refresh_token": "1//05nz9IRnR1e2JCgYIARAAGAUSNwF-L9Ir_zi7EkfNFJb02e5XxL3e4Qicf7BN708CjMo66DuPPMhkd09jS6lIra-zubUPFGORcyU",
    "quota_project_id": "dev-sd-lake",
    "universe_domain": "googleapis.com"
}
# --------------------------------------------------------------------------

# BigQuery & GCS scope is enough for almost every BQ call
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

creds = Credentials.from_authorized_user_info(adc_info, scopes=SCOPES)
bq = bigquery.Client(project="dev-sd-lake", credentials=creds)

# --- 2▸ helper -------------------------------------------------------------
def qdf(sql: str):
    """Run a query and return the results as a DataFrame."""
    return bq.query(sql).result().to_dataframe()

def explore_table(table_name: str):
    """Print count and first 2 rows of a table."""
    print(f"\n=== {table_name} ===")
    
    # Get first 2 rows
    sample_sql = f"SELECT * FROM pollen.{table_name} LIMIT 2"
    sample_df = qdf(sample_sql)
    print(f"Columns: {list(sample_df.columns)}")
    print("First 2 rows:")
    print(sample_df.to_string(index=False))
# --------------------------------------------------------------------------

# --- 3▸ explore your tables ------------------------------------------------
tables = [
    "product_listings",
    "products", 
    "product_categories",
    "product_subcategories",
    "sellers",
    "offers",
    "orders_level_1", 
    "orders_level_2"
]

for table in tables:
    explore_table(table)