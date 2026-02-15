from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    LayerNormalization,
    LSTM,
    Attention,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    revenue: np.ndarray
    variable_cost: np.ndarray
    net_profit_base: np.ndarray
    net_profit_corrected: Optional[np.ndarray] = None


@dataclass
class ScenarioResult:
    base: ForecastResult
    scenario: ForecastResult


class ProfitForecastingEnsemble:
    """
    Ансамбль LSTM + XGBoost для прогноза дневной чистой прибыли.

    - LSTM прогнозирует ряды выручки и переменных затрат.
    - XGBoost аппроксимирует остатки по чистой прибыли и исправляет прогноз.
    """

    def __init__(self, seq_len: int = 60, horizon: int = 30, random_state: int = 42):
        self.seq_len = seq_len
        self.horizon = horizon
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.lstm_model: Optional[Model] = None
        self.xgb_model: Optional[XGBRegressor] = None
        self.n_features: int = 0

    # ---------- подготовка признаков ----------

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)

        # календарные признаки
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # лаги и скользящие средние
        for col in ["revenue", "variable_cost", "net_profit"]:
            for lag in [1, 7, 14, 30]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            for w in [7, 14, 30]:
                df[f"{col}_ma_{w}"] = df[col].shift(1).rolling(w).mean()

        df = df.dropna()
        return df

    def _make_sequences(self, df: pd.DataFrame):
        df_feat = self._feature_engineering(df)

        feature_cols = [
            c
            for c in df_feat.columns
            if c not in ["date", "revenue", "variable_cost", "net_profit"]
        ]
        self.n_features = len(feature_cols)

        features = df_feat[feature_cols].values
        target_rev = df_feat["revenue"].values
        target_var = df_feat["variable_cost"].values
        target_profit = df_feat["net_profit"].values

        features_scaled = self.scaler.fit_transform(features)

        X_list, y_list, profit_list = [], [], []
        for i in range(self.seq_len, len(df_feat) - self.horizon + 1):
            X_seq = features_scaled[i - self.seq_len : i, :]
            y_h_rev = target_rev[i : i + self.horizon]
            y_h_var = target_var[i : i + self.horizon]
            y_h_profit = target_profit[i : i + self.horizon]

            X_list.append(X_seq)
            y_list.append(np.stack([y_h_rev, y_h_var], axis=1))  # [H, 2]
            profit_list.append(y_h_profit)

        if not X_list:
            raise ValueError(
                "Слишком мало данных для выбранного seq_len/horizon. "
                "Уменьши seq_len или горизонт прогноза."
            )

        X = np.stack(X_list)           # [N, L, F]
        y = np.stack(y_list)           # [N, H, 2]
        profit_arr = np.stack(profit_list)  # [N, H]

        return X, y, profit_arr, df_feat

    # ---------- архитектура LSTM ----------

    def _build_lstm(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="input_seq")

        x = Conv1D(filters=32, kernel_size=7, padding="same", activation="relu")(inp)
        x = LayerNormalization()(x)

        x = LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = LayerNormalization()(x)
        x = LSTM(32, return_sequences=True, dropout=0.2)(x)
        x = LayerNormalization()(x)

        # self-attention
        attn = Attention(use_scale=True)([x, x])

        # Пуллинг через слой Keras (а не tf.reduce_mean) — корректно для KerasTensor.
        context = GlobalAveragePooling1D()(attn)

        x = Dense(64, activation="relu")(context)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)

        # Два выхода: выручка и переменные затраты на горизонте H
        out_rev = Dense(self.horizon, name="rev_out")(x)
        out_var = Dense(self.horizon, name="var_out")(x)

        model = Model(inputs=inp, outputs=[out_rev, out_var])

        # Для двух выходов — список из двух лоссов
        model.compile(optimizer="adam", loss=["mae", "mae"])  # [loss_rev, loss_var][web:158][web:161]

        return model

    # ---------- обучение ----------

    def fit(self, df_daily: pd.DataFrame, epochs: int = 5) -> Dict[str, float]:
        """
        df_daily: DataFrame с колонками date, revenue, variable_cost, net_profit.
        """
        X, y, profit_arr, df_feat = self._make_sequences(df_daily)

        y_rev = y[:, :, 0]
        y_var = y[:, :, 1]

        (
            X_train,
            X_val,
            y_rev_tr,
            y_rev_val,
            y_var_tr,
            y_var_val,
            profit_tr,
            profit_val,
        ) = train_test_split(
            X,
            y_rev,
            y_var,
            profit_arr,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.lstm_model = self._build_lstm()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ]

        history = self.lstm_model.fit(
            X_train,
            [y_rev_tr, y_var_tr],
            validation_data=(X_val, [y_rev_val, y_var_val]),
            epochs=epochs,           # <= всего 5 эпох
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # Базовая метрика по чистой прибыли
        rev_val_pred, var_val_pred = self.lstm_model.predict(X_val, verbose=0)
        profit_val_pred = rev_val_pred - var_val_pred

        profit_true_mean = profit_val.mean(axis=1)
        profit_pred_mean = profit_val_pred.mean(axis=1)

        mae_base = mean_absolute_error(profit_true_mean, profit_pred_mean)

        # XGBoost по остаткам
        residuals = profit_true_mean - profit_pred_mean
        flat_features = X_val[:, -1, :]

        self.xgb_model = XGBRegressor(
            n_estimators=200,           # немного урезал, чтобы было быстрее
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )
        self.xgb_model.fit(flat_features, residuals)

        residual_pred = self.xgb_model.predict(flat_features)
        profit_corrected = profit_pred_mean + residual_pred
        mae_corrected = mean_absolute_error(profit_true_mean, profit_corrected)

        return {
            "mae_profit_base": float(mae_base),
            "mae_profit_corrected": float(mae_corrected),
            "best_val_loss": float(min(history.history["val_loss"])),
        }

    # ---------- прогноз ----------

    def forecast(self, df_daily: pd.DataFrame) -> ForecastResult:
        df_feat = self._feature_engineering(df_daily)

        feature_cols = [
            c
            for c in df_feat.columns
            if c not in ["date", "revenue", "variable_cost", "net_profit"]
        ]
        features = df_feat[feature_cols].values
        features_scaled = self.scaler.transform(features)

        last_seq = features_scaled[-self.seq_len :, :].reshape(
            1, self.seq_len, self.n_features
        )

        rev_pred, var_pred = self.lstm_model.predict(last_seq, verbose=0)
        rev_pred = rev_pred[0]
        var_pred = var_pred[0]
        profit_pred = rev_pred - var_pred

        profit_corrected = None
        if self.xgb_model is not None:
            flat_feat = last_seq[:, -1, :]
            residual_corr = self.xgb_model.predict(flat_feat)[0]
            profit_corrected = profit_pred + residual_corr

        last_date = df_feat["date"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.horizon,
            freq="D",
        )

        return ForecastResult(
            dates=future_dates,
            revenue=rev_pred,
            variable_cost=var_pred,
            net_profit_base=profit_pred,
            net_profit_corrected=profit_corrected,
        )

    # ---------- сценарный анализ ----------

    def forecast_scenario(
        self,
        df_daily: pd.DataFrame,
        marketing_multiplier: float = 1.0,
        price_multiplier: float = 1.0,
    ) -> ScenarioResult:
        """
        Меняем уровень цен и переменных расходов и смотрим,
        как это влияет на прогноз чистой прибыли.
        """
        base_res = self.forecast(df_daily)

        scen_rev = base_res.revenue * price_multiplier
        scen_var = base_res.variable_cost * marketing_multiplier
        scen_profit = scen_rev - scen_var

        if base_res.net_profit_corrected is not None:
            scen_profit_corr = (
                base_res.net_profit_corrected * price_multiplier
                - (base_res.variable_cost * marketing_multiplier)
            )
        else:
            scen_profit_corr = None

        scen = ForecastResult(
            dates=base_res.dates,
            revenue=scen_rev,
            variable_cost=scen_var,
            net_profit_base=scen_profit,
            net_profit_corrected=scen_profit_corr,
        )

        return ScenarioResult(base=base_res, scenario=scen)

