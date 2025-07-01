import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import matplotlib.pyplot as plt
import os
import contextlib
from plot_metric.functions import BinaryClassification
from utils.fern_static_variables import sales_prob_train_cols, sales_prob_cat_cols, sales_prob_target_col, sales_prob_price_model_params,\
    domestic_export_train_cols, domestic_export_cat_cols, domestic_export_target_col, domestic_export_price_model_params, time_at_a_price_model_params,\
    time_at_a_price_train_cols, time_at_a_price_cat_cols, time_at_a_price_target_col, amount_at_a_price_model_params, amount_at_a_price_train_cols,\
    amount_at_a_price_cat_cols, amount_at_a_price_target_col


def train_sales_prob_price_model(total_inv_after_sampled):
    print('Training sales_prob_price_model...')
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
    print('Training domestic_export_model...')
    print(total_inv_after_sampled[domestic_export_train_cols].duplicated().sum())
    total_inv_after_sampled = total_inv_after_sampled.drop_duplicates(subset = domestic_export_train_cols)
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled['product_category'].isnull()]

    X = total_inv_after_sampled[domestic_export_train_cols]
    y = total_inv_after_sampled[domestic_export_target_col]

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
    train_pool = Pool(X_train, y_train, cat_features=domestic_export_cat_cols, weight=[class_weights_dict[label] for label in y_train])
    val_pool = Pool(X_val, y_val, cat_features=domestic_export_cat_cols)

    domestic_export_price_model = CatBoostClassifier(**domestic_export_price_model_params, iterations = 1000)
    domestic_export_price_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    pred_probs = domestic_export_price_model.predict_proba(X_val)
    
    return domestic_export_price_model, y_val, pred_probs


def train_time_at_a_price_model(total_inv_after_sampled):
    print('Training time_at_a_price_model...')
    print(total_inv_after_sampled[time_at_a_price_train_cols].duplicated().sum())
    total_inv_after_sampled = total_inv_after_sampled.drop_duplicates(subset = time_at_a_price_train_cols)
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled['product_category'].isnull()]
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled[time_at_a_price_target_col].isnull()]

    total_inv_after_sampled[time_at_a_price_target_col] = np.where(total_inv_after_sampled[time_at_a_price_target_col] > 90,
                                                                    np.random.uniform(90,120), total_inv_after_sampled[time_at_a_price_target_col])
    total_inv_after_sampled[time_at_a_price_target_col] = np.log1p(total_inv_after_sampled[time_at_a_price_target_col])

    X = total_inv_after_sampled[time_at_a_price_train_cols]
    y = total_inv_after_sampled[time_at_a_price_target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(X_train.shape, X_val.shape)

    time_at_a_price_model = CatBoostRegressor(**time_at_a_price_model_params, iterations = 1000)
    time_at_a_price_model.fit(X_train, y_train, cat_features=time_at_a_price_cat_cols, eval_set=(X_val, y_val), verbose=True)

    time_preds = time_at_a_price_model.predict(X_val)
    time_preds = np.expm1(time_preds)
    time_at_a_price_inverse_y_val = np.expm1(y_val)

    return time_at_a_price_model, time_at_a_price_inverse_y_val, time_preds


def train_amount_at_a_price_model(total_inv_after_sampled):
    print('Training amount_at_a_price_model...')
    print(total_inv_after_sampled[amount_at_a_price_train_cols].duplicated().sum())
    total_inv_after_sampled = total_inv_after_sampled.drop_duplicates(subset = amount_at_a_price_train_cols)
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled['product_category'].isnull()]
    total_inv_after_sampled = total_inv_after_sampled[~total_inv_after_sampled[amount_at_a_price_target_col].isnull()]

    total_inv_after_sampled[amount_at_a_price_target_col] = np.where(total_inv_after_sampled[amount_at_a_price_target_col] > total_inv_after_sampled.total_units.quantile(0.98),
                                                    total_inv_after_sampled.total_units.quantile(0.98), total_inv_after_sampled[amount_at_a_price_target_col])
    total_inv_after_sampled[amount_at_a_price_target_col] = np.log1p(total_inv_after_sampled[amount_at_a_price_target_col])

    X = total_inv_after_sampled[amount_at_a_price_train_cols]
    y = total_inv_after_sampled[amount_at_a_price_target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(X_train.shape, X_val.shape)

    amount_at_a_price_model = CatBoostRegressor(**amount_at_a_price_model_params, iterations = 1000)
    amount_at_a_price_model.fit(X_train, y_train, cat_features=amount_at_a_price_cat_cols, eval_set=(X_val, y_val), verbose=True)

    amount_preds = amount_at_a_price_model.predict(X_val)
    amount_preds = np.expm1(amount_preds)
    amount_at_a_price_inverse_y_val = np.expm1(y_val)

    return amount_at_a_price_model, amount_at_a_price_inverse_y_val, amount_preds


# tune the threshold
def plot_metrics_report(y_val, pred_probs, t, model_name, output_dir="metrics_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize BinaryClassification object
    bc = BinaryClassification(y_val, pred_probs[:, 1], labels=["Not Sellable", "Sellable"])

    # --- Create and save the metrics plot ---
    fig = plt.figure(figsize=(15, 10))
    plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    bc.plot_roc_curve(threshold=t)

    plt.subplot2grid((2,6), (0,2), colspan=2)
    bc.plot_precision_recall_curve(threshold=t)

    plt.subplot2grid((2,6), (0,4), colspan=2)
    bc.plot_class_distribution(pal_colors=['r','g','m','k'], threshold=t)

    plt.subplot2grid((2,6), (1,1), colspan=2)
    bc.plot_confusion_matrix(threshold=t)

    # Save plot image
    fig_path = os.path.join(output_dir, f"{model_name}_metrics_plot.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)

    # --- Redirect print_report() output to a file ---
    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        with contextlib.redirect_stdout(f):
            bc.print_report(threshold=t)

    print(f"[✓] Saved metrics and classification report for: {model_name}")