"""
Script de despliegue del modelo MLP (Red Neuronal) para predicción de casos de dengue
SOLO VARIABLES EPIDEMIOLÓGICAS CON REZAGO
Generado automáticamente - 2026-09-16 09:09:05

Este script permite cargar el modelo entrenado y realizar predicciones
sobre nuevos datos con el mismo formato que los datos de entrenamiento.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class DengueModelDeployment:
    """
    Clase para desplegar el modelo MLP de predicción de casos de dengue
    usando SOLO variables epidemiológicas con rezago.
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.model_dir, 'mlp_model.pkl'), 'rb') as f:
            self.mlp_model = pickle.load(f)

        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
            self.selected_features = config['selected_features']
            self.X_augmented_columns = config['X_augmented_columns']
            self.best_params = config.get('best_params', {})
            self.best_architecture = config.get('best_architecture', {})

        print("✅ Modelo MLP (epidemiológico) cargado exitosamente")
        print(f"   Características seleccionadas: {len(self.selected_features)}")
        print(f"   Arquitectura: {self.best_architecture.get('hidden_layers', 'No especificada')}")

    def _get_epidemiological_columns(self, X):
        epi_cols = []
        exclude_exact = {{'fecha', 'date', 'datetime', 'timestamp',
                          'año', 'ano', 'year', 'semana_epi', 'semana',
                          'casos_dengue'}}
        for col in X.columns:
            col_lower = col.lower()
            if col_lower in exclude_exact:
                continue
            if col.startswith('casos_dengue_lag_'):
                epi_cols.append(col)
        try:
            epi_cols = sorted(epi_cols, key=lambda c: int(c.split('_')[-1]))
        except Exception:
            pass
        return epi_cols

    def _create_lag_aggregates(self, X):
        lag_aggs = pd.DataFrame(index=X.index)
        dengue_lag_cols = [c for c in X.columns if c.startswith('casos_dengue_lag_')]
        if len(dengue_lag_cols) >= 3:
            lag_data = X[dengue_lag_cols]
            lag_aggs['dengue_lag_mean'] = lag_data.mean(axis=1)
            lag_aggs['dengue_lag_std'] = lag_data.std(axis=1)
            lag_aggs['dengue_lag_sum'] = lag_data.sum(axis=1)
            lag_aggs['dengue_lag_max'] = lag_data.max(axis=1)
            lag_aggs['dengue_lag_min'] = lag_data.min(axis=1)
            lag_aggs['dengue_lag_median'] = lag_data.median(axis=1)
            lag_aggs['dengue_lag_range'] = (lag_aggs['dengue_lag_max'] -
                                             lag_aggs['dengue_lag_min'])
            lag_aggs['dengue_lag_cv'] = (lag_aggs['dengue_lag_std'] /
                                          (lag_aggs['dengue_lag_mean'] + 1e-6))
        for window, name in [(3, 'short'), (6, 'mid'), (12, 'long')]:
            cols = ['casos_dengue_lag_' + str(i) for i in range(1, window + 1)]
            cols = [c for c in cols if c in X.columns]
            if len(cols) == window:
                window_data = X[cols]
                lag_aggs['dengue_' + name + '_mean'] = window_data.mean(axis=1)
                lag_aggs['dengue_' + name + '_std'] = window_data.std(axis=1)
                lag_aggs['dengue_' + name + '_max'] = window_data.max(axis=1)
                lag_aggs['dengue_' + name + '_sum'] = window_data.sum(axis=1)
        return lag_aggs

    def _create_trend_features(self, X):
        trend_features = pd.DataFrame(index=X.index)
        for i in range(1, 12):
            col_curr = 'casos_dengue_lag_' + str(i)
            col_next = 'casos_dengue_lag_' + str(i + 1)
            if col_curr in X.columns and col_next in X.columns:
                trend_features['dengue_diff_' + str(i) + '_' + str(i + 1)] = (
                    X[col_curr] - X[col_next])
        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_4' in X.columns:
            trend_features['dengue_trend_short'] = (
                X['casos_dengue_lag_1'] - X['casos_dengue_lag_4'])
        if 'casos_dengue_lag_4' in X.columns and 'casos_dengue_lag_8' in X.columns:
            trend_features['dengue_trend_mid'] = (
                X['casos_dengue_lag_4'] - X['casos_dengue_lag_8'])
        if 'casos_dengue_lag_8' in X.columns and 'casos_dengue_lag_12' in X.columns:
            trend_features['dengue_trend_long'] = (
                X['casos_dengue_lag_8'] - X['casos_dengue_lag_12'])
        if ('casos_dengue_lag_1' in X.columns and
            'casos_dengue_lag_2' in X.columns and
                'casos_dengue_lag_3' in X.columns):
            trend_features['dengue_acceleration'] = (
                X['casos_dengue_lag_1'] - 2 * X['casos_dengue_lag_2'] +
                X['casos_dengue_lag_3'])
        return trend_features

    def _create_ratio_features(self, X):
        ratio_features = pd.DataFrame(index=X.index)
        ratios = [
            ('casos_dengue_lag_1', 'casos_dengue_lag_2', 'dengue_ratio_1_2'),
            ('casos_dengue_lag_1', 'casos_dengue_lag_4', 'dengue_ratio_1_4'),
            ('casos_dengue_lag_1', 'casos_dengue_lag_12', 'dengue_ratio_1_12'),
            ('casos_dengue_lag_2', 'casos_dengue_lag_4', 'dengue_ratio_2_4'),
            ('casos_dengue_lag_4', 'casos_dengue_lag_12', 'dengue_ratio_4_12'),
        ]
        for col1, col2, new_col in ratios:
            if col1 in X.columns and col2 in X.columns:
                ratio_features[new_col] = X[col1] / (X[col2] + 1.0)
        return ratio_features

    def _create_rolling_features(self, X):
        rolling_features = pd.DataFrame(index=X.index)
        if 'semana_epi' in X.columns:
            rolling_features['week_sin'] = np.sin(2 * np.pi * X['semana_epi'] / 52)
            rolling_features['week_cos'] = np.cos(2 * np.pi * X['semana_epi'] / 52)
        return rolling_features

    def _create_polynomial_features(self, X):
        important_lags = ['casos_dengue_lag_1', 'casos_dengue_lag_2',
                          'casos_dengue_lag_4', 'casos_dengue_lag_12']
        poly_features = pd.DataFrame(index=X.index)
        for var in important_lags:
            if var in X.columns:
                poly_features[var + '_squared'] = X[var] ** 2
                poly_features[var + '_sqrt'] = np.sqrt(np.abs(X[var]))
        return poly_features

    def _create_interaction_features(self, X):
        interaction_data = pd.DataFrame(index=X.index)
        interactions = [
            ('casos_dengue_lag_1', 'casos_dengue_lag_2', 'lag1_lag2'),
            ('casos_dengue_lag_1', 'casos_dengue_lag_4', 'lag1_lag4'),
            ('casos_dengue_lag_2', 'casos_dengue_lag_4', 'lag2_lag4'),
            ('casos_dengue_lag_4', 'casos_dengue_lag_12', 'lag4_lag12'),
            ('casos_dengue_lag_1', 'casos_dengue_lag_12', 'lag1_lag12'),
        ]
        for col1, col2, new_col in interactions:
            if col1 in X.columns and col2 in X.columns:
                interaction_data[new_col] = X[col1] * X[col2]
        return interaction_data

    def _augment_features(self, X):
        epi_cols = self._get_epidemiological_columns(X)
        X_epi = X[epi_cols].copy()
        X_epi = X_epi.dropna(axis=1, how='all')
        features_list = [X_epi]

        lag_aggs = self._create_lag_aggregates(X_epi)
        if not lag_aggs.empty:
            features_list.append(lag_aggs)
        trend_features = self._create_trend_features(X_epi)
        if not trend_features.empty:
            features_list.append(trend_features)
        ratio_features = self._create_ratio_features(X_epi)
        if not ratio_features.empty:
            features_list.append(ratio_features)
        poly_features = self._create_polynomial_features(X_epi)
        if not poly_features.empty:
            features_list.append(poly_features)
        interaction_features = self._create_interaction_features(X_epi)
        if not interaction_features.empty:
            features_list.append(interaction_features)
        rolling_features = self._create_rolling_features(X)
        if not rolling_features.empty:
            features_list.append(rolling_features)

        X_augmented = pd.concat(features_list, axis=1)
        X_augmented = X_augmented.replace([np.inf, -np.inf], np.nan)
        X_augmented = X_augmented.fillna(X_augmented.mean())
        X_augmented = X_augmented.fillna(0)
        return X_augmented

    def predict(self, X):
        X_augmented = self._augment_features(X)
        for col in self.X_augmented_columns:
            if col not in X_augmented.columns:
                X_augmented[col] = 0
        X_augmented = X_augmented[self.X_augmented_columns]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_augmented),
            columns=X_augmented.columns,
            index=X_augmented.index
        )
        X_final = X_scaled[self.selected_features]
        predictions = self.mlp_model.predict(X_final)
        return predictions

    def predict_and_save(self, X, output_path=None):
        predictions = self.predict(X)
        results_df = X.copy()
        results_df['prediccion_casos_dengue'] = predictions
        if output_path:
            results_df.to_excel(output_path, index=False)
            print(f"✅ Predicciones guardadas en: {output_path}")
        return results_df


if __name__ == "__main__":
    print("="*60)
    print("SCRIPT DE DESPLIEGUE - MODELO EPIDEMIOLÓGICO DENGUE")
    print("="*60)

    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\6_redes_neuronales\1_red_neuronal_mlp\2_datos\1_raw\2_90_10\modelo_mlp_epidemiologico_guardado"
    model = DengueModelDeployment(model_dir)

    print("\nCargando datos de prueba...")
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\6_redes_neuronales\1_red_neuronal_mlp\2_datos\1_raw\2_90_10\2_meteo_epi_2021-2026_1_rezagos_meteo_epi_test_90_10.xlsx"
    test_data = pd.read_excel(test_path)

    X_test = test_data.drop('casos_dengue', axis=1)

    print("Realizando predicciones...")
    predictions = model.predict(X_test)

    print(f"Predicciones generadas para {len(predictions)} muestras")
    print(f"Estadísticas de predicciones:")
    print(f"  Media: {np.mean(predictions):.2f}")
    print(f"  Mediana: {np.median(predictions):.2f}")
    print(f"  Mínimo: {np.min(predictions):.2f}")
    print(f"  Máximo: {np.max(predictions):.2f}")

    output_path = os.path.join(model_dir, "predicciones_ejemplo.xlsx")
    model.predict_and_save(X_test, output_path)
