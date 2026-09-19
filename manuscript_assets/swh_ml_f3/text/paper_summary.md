
This study evaluated a cascade machine-learning pipeline for a solar water-heating dataset with 372 cleaned observations, using a 75/25 hold-out split and 5-fold cross-validation on the training fold only.

Stage 1 predicted outlet temperature (Tout) using a tuned RandomForest model, with hold-out RMSE 0.5753, MAE 0.3319, R2 0.9913, and adjusted R2 0.9907.

Stage 2 used the predicted Tout signal as an additional feature for water-outlet prediction and selected a tuned RandomForest model, achieving hold-out RMSE 0.1922, MAE 0.1198, R2 0.9999, and adjusted R2 0.9999.

Residual diagnostics were used to inspect bias and dispersion, and the stage-1-to-stage-2 error correlation was -0.0227, indicating how first-stage prediction error propagated through the cascade.

Dynamic correlation-based feature selection was applied inside the training fold to keep the workflow leakage-safe and reproducible.
