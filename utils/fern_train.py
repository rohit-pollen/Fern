import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from plot_metric.functions import BinaryClassification

from utils.fern_static_variables import sales_prob_train_cols, sales_prob_cat_cols, sales_prob_target_col, sales_prob_price_model_params,\
    domestic_export_train_cols, domestic_export_cat_cols, domestic_export_target_col, domestic_export_price_model_params


def train_sales_prob_price_model(total_inv_after_sampled):

    print(total_inv_after_sampled[sales_prob_train_cols].duplicated().sum())
    total_inv_after_sampled = total_inv_after_sampled.drop_duplicates(subset = sales_prob_train_cols)
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled['product_category'].isnull()]
    print(total_inv_after_sampled.sellability.value_counts())

    X = total_inv_after_sampled[sales_prob_train_cols]
    y = total_inv_after_sampled[sales_prob_target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(X_train.shape, X_val.shape)

    # 4. Compute class weights (inverse of class frequencies)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # 5. Prepare CatBoost Pools
    train_pool = Pool(X_train, y_train, cat_features=sales_prob_cat_cols, weight=[class_weights_dict[label] for label in y_train])
    val_pool = Pool(X_val, y_val, cat_features=sales_prob_cat_cols)

    sales_prob_price_model = CatBoostClassifier(**sales_prob_price_model_params, iterations = 1000)
    sales_prob_price_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    pred_probs = sales_prob_price_model.predict_proba(X_val)
    return sales_prob_price_model, y_val, pred_probs

def train_domestic_export_model(total_inv_after_sampled):
    print(total_inv_after_sampled[domestic_export_train_cols].duplicated().sum())
    total_inv_after_sampled = total_inv_after_sampled.drop_duplicates(subset = domestic_export_train_cols)
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled['product_category'].isnull()]

    X = total_inv_after_sampled[domestic_export_train_cols]
    y = total_inv_after_sampled[domestic_export_target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Compute class weights (inverse of class frequencies)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # 5. Prepare CatBoost Pools
    train_pool = Pool(X_train, y_train, cat_features=domestic_export_cat_cols, weight=[class_weights_dict[label] for label in y_train])
    val_pool = Pool(X_val, y_val, cat_features=domestic_export_cat_cols)

    domestic_export_price_model = CatBoostClassifier(**domestic_export_price_model_params, iterations = 1000)
    domestic_export_price_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    pred_probs = domestic_export_price_model.predict_proba(X_val)
    
    return domestic_export_price_model, y_val, pred_probs

# tune the threshold
def plot_metrics_report(y_val, pred_probs, t):
    bc = BinaryClassification(y_val, pred_probs[:, 1], labels=["Not Sellable", "Sellable"])
    # Figures
    plt.figure(figsize=(15,10))
    plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    bc.plot_roc_curve(threshold = t)
    plt.subplot2grid((2,6), (0,2), colspan=2)
    bc.plot_precision_recall_curve(threshold = t)
    plt.subplot2grid((2,6), (0,4), colspan=2)
    bc.plot_class_distribution(pal_colors=['r','g','m','k'], threshold = t)
    plt.subplot2grid((2,6), (1,1), colspan=2)
    a = bc.plot_confusion_matrix(threshold = t)

    plt.show()
    bc.print_report(threshold = t)