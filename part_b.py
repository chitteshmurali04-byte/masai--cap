    predictions_match = bool(np.array_equal(selected_model.predict(X_test), reload_pred))
    if not predictions_match:
        raise AssertionError("Reloaded model predictions differ from original predictions.")

    # Premium is a post-hoc whole-window flag from task 3; use only to filter
    # the illustrative output list, never as a model feature.
    segments = pd.read_csv(OUTPUTS / "customer_segments.csv")
    premium_ids = set(segments.loc[segments["is_premium"].astype(str).str.lower().isin(
        ["true", "1"]), "customer_id"])
    premium_mask = id_test.isin(premium_ids).to_numpy()
    probabilities = selected_model.predict_proba(X_test)[:, 1]
    retention = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "lapse_probability": probabilities,
        "actually_lapsed": y_test.to_numpy(),
        "is_premium": premium_mask,
    })
    retention = retention.loc[retention["is_premium"]].sort_values(
        "lapse_probability", ascending=False
    ).head(10)
    retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
        OUTPUTS / "retention_list.csv", index=False
    )
    retention_true_lapsed = int(retention["actually_lapsed"].sum())

    cohort_summary = {
        "cohort_customers": int(len(cohort)),
        "lapsed_count": int(y.sum()),
        "lapsed_percentage": float(y.mean() * 100),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    summary = {
        "cohort": cohort_summary,
        "baseline": baseline,
        "logistic_regression": {
            "best_params": log_grid.best_params_,
            "best_cv_f1": float(log_grid.best_score_),
            "test": log_test,
        },
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "best_cv_f1": float(rf_grid.best_score_),
            "test": rf_test,
        },
        "selected_model": selected_name,
        "selected_test_metrics": selected_metrics,
        "balanced_logistic_regression": {
            "best_params": balanced_log.best_params_,
            "best_cv_f1": float(balanced_log.best_score_),
            "test": balanced_metrics,
            "balanced_training_class_counts": {
                str(k): int(v) for k, v in y_bal.value_counts().to_dict().items()
            },
        },
        "scaler_frequency_mean_train": learned_frequency_mean,
        "frequency_mean_full_cohort": full_frequency_mean,
        "missing_values_by_feature_before_pipeline_imputation": {
            str(k): int(v) for k, v in missing_counts.items()
        },
        "rf_top_five_features": top_features,
        "logistic_top_three_coefficients": top_coefficients,
        "reloaded_predictions_exact_match": predictions_match,
        "retention_list_rows": int(len(retention)),
        "retention_list_actually_lapsed": retention_true_lapsed,
        "premium_flag_limitation": (
            "Premium is a post-hoc whole-window RFM label that includes the "
            "September lapse-label window. The retention list is illustrative, "
            "not a leakage-free evaluation."
        ),
    }
    return summary


def main():
    orders, customers, products = read_inputs()

    rfm = make_rfm(orders)
    rfm_summary = describe_rfm(rfm)
    segments, k_table, profiles, pca_ratio, chosen_k, premium_id = cluster_customers(rfm)

    bundle = build_lapse_dataset(orders, customers)
    model_summary = train_models(bundle)

    summary = {
        "rfm": rfm_summary,
        "kmeans": {
            "metrics": k_table.to_dict(orient="records"),
            "chosen_k": chosen_k,
            "profiles": profiles.to_dict(orient="records"),
            "pca_explained_variance_ratio": pca_ratio,
            "premium_segment_id": premium_id,
            "segment_rows": int(len(segments)),
        },
        **model_summary,
    }

    with open(OUTPUTS / "part_b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== PART B: RFM SUMMARY ===")
    print(json.dumps(rfm_summary, indent=2))
    print("\n=== K-MEANS METRICS (K=2..8) ===")
    print(k_table.to_string(index=False))
    print(f"\nChosen K: {chosen_k}")
    print("\n=== SEGMENT PROFILES ===")
    print(profiles.to_string(index=False))
    print("\nPCA explained variance ratio:", pca_ratio)
    print("\n=== LAPSE MODEL RESULTS ===")
    print(json.dumps(model_summary, indent=2, default=str))
    print("\nCreated:")
    print("  outputs/customer_segments.csv")
    print("  outputs/lapse_model.joblib")
    print("  outputs/retention_list.csv")
    print("  outputs/part_b_summary.json")


if __name__ == "__main__":
    main()
