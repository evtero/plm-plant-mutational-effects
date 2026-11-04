from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class Train:
    name = "train"

    def __init__(
        self,
        method_params: dict,
        output_dir: Path,
    ) -> None:
        self.method_params = method_params
        self.output_dir = Path(output_dir)

        self.train_features_path = Path(method_params["train_features_path"])
        self.test_features_path = Path(method_params["test_features_path"])

        self.selection_metric: str = method_params.get("selection_metric", "f1_weighted")
        self.n_splits: int = int(method_params.get("cv_n_splits", 5))
        self.random_state: int = int(method_params.get("random_state", 42))
        self.group_col: str = method_params.get("group_col", "family_id")

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = self.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train, meta_train = self._load_features(self.train_features_path)
        X_test, y_test, meta_test = self._load_features(self.test_features_path)

        if self.group_col not in meta_train.columns:
            raise ValueError(
                f"meta_train must contain group column '{self.group_col}' "
                f"to avoid CV leakage by family"
            )

        groups_train = meta_train[self.group_col].astype(str).values

        logger.info(
            "[%s] train=%s | test=%s | X_dim=%d | group_col=%s | n_groups=%d",
            self.name,
            X_train.shape,
            X_test.shape,
            X_train.shape[1],
            self.group_col,
            len(set(groups_train)),
        )

        self._validate_train_test(X_train, y_train, X_test, y_test, groups_train)

        cv = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info("[%s] CV fold summary:", self.name)

        
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train, groups_train), start=1
        ):
            y_tr = y_train[train_idx]
            y_va = y_train[val_idx]
            groups_tr = meta_train.iloc[train_idx][self.group_col]
            groups_va = meta_train.iloc[val_idx][self.group_col]

            logger.info(
                "[Fold %d] train=%d val=%d | train_groups=%d val_groups=%d | "
                "train_effect=%s | val_effect=%s",
                fold,
                len(train_idx),
                len(val_idx),
                groups_tr.nunique(),
                groups_va.nunique(),
                pd.Series(y_tr).value_counts().to_dict(),
                pd.Series(y_va).value_counts().to_dict(),
            )

            overlap = set(groups_tr) & set(groups_va)
            logger.info("[Fold %d] overlapping families = %d", fold, len(overlap))
            if overlap:
                raise RuntimeError(f"Fold {fold}: group leakage detected")

        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
            "f1_class_0": make_scorer(f1_score, pos_label=0, zero_division=0),
            "f1_class_1": make_scorer(f1_score, pos_label=1, zero_division=0),
            "roc_auc": "roc_auc",
        }

        best_model = None
        best_score = -float("inf")
        best_model_name = None
        summary_rows: list[dict] = []
        grid_results: list[pd.DataFrame] = []


        tracker = EmissionsTracker(
            output_dir=str(models_dir),
            project_name="training",
            output_file="emissions_training.csv",
            log_level="error",
        )
        tracker.start()
        try:

            for grid in self._param_grid():
                model_name = grid["model"][0].__class__.__name__

                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", grid["model"][0]),
                    ]
                )

                gs = GridSearchCV(
                    estimator=pipeline,
                    param_grid=grid,
                    cv=cv,
                    scoring=scoring,
                    refit=self.selection_metric,
                    n_jobs=16,
                    return_train_score=True,
                    verbose=1,
                )

                gs.fit(X_train, y_train, groups=groups_train)

                cv_df = pd.DataFrame(gs.cv_results_)
                cv_df["model_name"] = model_name
                grid_results.append(cv_df)

                best_idx = gs.best_index_
                row: dict = {
                    "model": model_name,
                    "best_params": json.dumps(gs.best_params_, default=self._to_jsonable),
                    "cv_best_score": float(gs.best_score_),
                    "is_best": False,
                }
                for metric in scoring:
                    row[f"cv_{metric}_mean"] = float(
                        gs.cv_results_[f"mean_test_{metric}"][best_idx]
                    )
                    row[f"cv_{metric}_std"] = float(
                        gs.cv_results_[f"std_test_{metric}"][best_idx]
                    )

                if gs.best_score_ > best_score:
                    best_score = float(gs.best_score_)
                    best_model = gs.best_estimator_
                    best_model_name = model_name
                    row["is_best"] = True

                summary_rows.append(row)
        finally:
            tracker.stop()

        pd.DataFrame(summary_rows).to_csv(models_dir / "cv_metrics_summary.csv", index=False)
        pd.concat(grid_results, ignore_index=True).to_csv(
            models_dir / "grid_search_results.csv", index=False
        )

        summary_df = pd.DataFrame(summary_rows)

        summary_df.to_csv(
            models_dir / "cv_metrics_summary.csv",
            index=False,
        )

        summary_metrics_fold_rows = []

        for cv_df in grid_results:
            model_name = cv_df["model_name"].iloc[0]

            best_idx = cv_df[f"mean_test_{self.selection_metric}"].idxmax()
            best_row = cv_df.loc[best_idx]

            for metric in [
                "f1_weighted",
                "f1_macro",
                "f1_class_0",
                "f1_class_1",
            ]:
                summary_metrics_fold_rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "train_mean": float(best_row[f"mean_train_{metric}"]),
                        "train_std": float(best_row[f"std_train_{metric}"]),
                        "val_mean": float(best_row[f"mean_test_{metric}"]),
                        "val_std": float(best_row[f"std_test_{metric}"]),
                        "is_best": model_name == best_model_name,
                    }
                )

        summary_metrics_fold = pd.DataFrame(summary_metrics_fold_rows)

        summary_metrics_fold.to_csv(
            models_dir / "summary_metrics_fold.tsv",
            sep="\t",
            index=False,
        )

        if best_model is None:
            raise RuntimeError("No model was trained")

        tracker = EmissionsTracker(
        output_dir=str(models_dir),
        project_name="testing",
        output_file="emissions_testing.csv",
        log_level="error",
        )
        tracker.start()
        try:
            y_pred = best_model.predict(X_test)
            test_result: dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            }

            y_proba = None
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                test_result["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        finally:
            tracker.stop()
        
    
        report = classification_report(y_test, y_pred, output_dict=True)

        model_path = models_dir / f"bestmodel_{best_model_name}.pkl"
        joblib.dump(best_model, model_path)

        test_pred_df = meta_test.copy()
        test_pred_df["y_true"] = y_test
        test_pred_df["y_pred"] = y_pred
        if y_proba is not None:
            test_pred_df["y_proba"] = y_proba

        test_predictions_path = models_dir / "test_predictions.tsv"
        test_pred_df.to_csv(test_predictions_path, sep="\t", index=False)

        result = {
            "best_model": best_model_name,
            "selection_metric": self.selection_metric,
            "best_cv_score": best_score,
            "test_metrics": test_result,
            "classification_report": report,
            "model_path": str(model_path),
            "metrics_path": str(models_dir / "cv_metrics_summary.csv"),
            "grid_path": str(models_dir / "grid_search_results.csv"),
            "test_predictions_path": str(test_predictions_path),
            "train_features_path": str(self.train_features_path),
            "test_features_path": str(self.test_features_path),
            "group_col": self.group_col,
            "n_train_groups": len(set(groups_train)),
        }

        result_path = models_dir / "train_result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=self._to_jsonable)

        logger.info("[%s] done | result=%s", self.name, result_path)
        return result

    def _load_features(self, path: Path):
        with path.open("rb") as f:
            obj = pickle.load(f)

        X = np.asarray(obj["X"])
        y = np.asarray(obj["y"]).astype(int)
        meta = obj["meta"]

        if not isinstance(meta, pd.DataFrame):
            meta = pd.DataFrame(meta)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D in {path}, got shape {X.shape}")

        if len(X) != len(y) or len(X) != len(meta):
            raise ValueError(
                f"Length mismatch in {path}: X={len(X)} y={len(y)} meta={len(meta)}"
            )

        return X, y, meta

    def _validate_train_test(self, X_train, y_train, X_test, y_test, groups_train):
        if len(X_train) == 0:
            raise ValueError("Empty training set")
        if len(X_test) == 0:
            raise ValueError("Empty test set")
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set has only one class")
        if len(np.unique(y_test)) < 2:
            logger.warning("[%s] Test set has only one class", self.name)

        n_groups = len(set(groups_train))
        if n_groups < self.n_splits:
            raise ValueError(
                f"Not enough groups for StratifiedGroupKFold: "
                f"n_groups={n_groups}, n_splits={self.n_splits}"
            )

        class_counts = pd.Series(y_train).value_counts().to_dict()
        logger.info("[%s] y_train class_counts=%s", self.name, class_counts)

    def _param_grid(self) -> list[dict]:
        return [
            {
                "model": [SVC(probability=True, class_weight="balanced")],
                "model__C": [1, 3],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale", "auto"],
            },
            {
                "model": [
                    RandomForestClassifier(
                        random_state=self.random_state,
                        n_jobs=16,
                        class_weight="balanced",
                    )
                ],
                "model__n_estimators": [50, 100, 200],
                "model__max_features": ["sqrt"],
                "model__min_samples_split": [10, 12],
                "model__min_samples_leaf": [4, 6],
                "model__max_depth": [7, 10],
            },
            {
                "model": [
                    XGBClassifier(
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=self.random_state,
                        scale_pos_weight=1,
                    )
                ],
                "model__n_estimators": [50, 100],
                "model__max_depth": [5, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
            },
        ]

    @staticmethod
    def _to_jsonable(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return str(obj)

