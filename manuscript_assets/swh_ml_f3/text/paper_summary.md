
This study evaluated a cascade machine-learning pipeline for a solar water-heating dataset with 38 cleaned observations, using a 75/25 hold-out split and 5-fold cross-validation on the training fold only.

Stage 1 predicted outlet temperature (Tout) using a tuned AdaBoost model, with hold-out RMSE 2.1230, MAE 1.5791, R2 0.8590, and adjusted R2 0.5769.

Stage 2 used the predicted Tout signal as an additional feature for water-outlet prediction and selected a tuned RandomForest model, achieving hold-out RMSE 0.0606, MAE 0.0413, R2 0.9990, and adjusted R2 0.9970.

Residual diagnostics were used to inspect bias and dispersion, and the stage-1-to-stage-2 error correlation was 0.1544, indicating how first-stage prediction error propagated through the cascade.

Dynamic correlation-based feature selection was applied inside the training fold to keep the workflow leakage-safe and reproducible.
