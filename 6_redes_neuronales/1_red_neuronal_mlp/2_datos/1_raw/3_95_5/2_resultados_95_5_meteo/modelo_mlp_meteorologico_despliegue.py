"""
Script de despliegue del modelo MLP (Red Neuronal) para predicción de casos de dengue
SOLO VARIABLES METEOROLÓGICAS
Generado automáticamente - 2026-09-16 08:46:55

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

# Variables meteorológicas utilizadas
METEOROLOGICAL_VARS = [
    'temp', 'temp_max', 'temp_min',
    'hum_esp', 'hum_rel',
    'prec', 'dias_lluvia',
    'vel_vi', 'vel_vi_max', 'vel_vi_min'
]

METEOROLOGICAL_LAG_VARS = METEOROLOGICAL_VARS


class DengueModelDeployment:
    """
    Clase para desplegar el modelo MLP de predicción de casos de dengue
    usando SOLO variables meteorológicas.
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

        print("✅ Modelo MLP (meteorológico) cargado exitosamente")
        print(f"   Características seleccionadas: {len(self.selected_features)}")
        print(f"   Arquitectura: {self.best_architecture.get('hidden_layers', 'No especificada')}")

    def _get_meteorological_columns(self, X):
        meteo_cols = []
        exclude_exact = {{'fecha', 'date', 'datetime', 'timestamp',
                          'año', 'ano', 'year', 'semana_epi', 'semana',
                          'casos_dengue'}}

        for col in X.columns:
            col_lower = col.lower()
            if col_lower in exclude_exact:
                continue
            if col in METEOROLOGICAL_VARS:
                meteo_cols.append(col)
                continue
            for base_var in METEOROLOGICAL_LAG_VARS:
                if col.startswith(base_var + '_lag_'):
                    meteo_cols.append(col)
                    break
        return meteo_cols

    def _create_interaction_features(self, X):
        interaction_data = pd.DataFrame(index=X.index)
        interactions = [
            ('temp', 'hum_rel', 'temp_hum_rel'),
            ('temp_max', 'hum_rel', 'temp_max_hum_rel'),
            ('temp_min', 'hum_rel', 'temp_min_hum_rel'),
            ('prec', 'temp', 'prec_temp'),
            ('prec', 'temp_max', 'prec_temp_max'),
            ('prec', 'temp_min', 'prec_temp_min'),
            ('prec', 'dias_lluvia', 'prec_dias_lluvia'),
            ('hum_rel', 'prec', 'hum_rel_prec'),
            ('hum_esp', 'prec', 'hum_esp_prec'),
            ('vel_vi', 'temp', 'vel_vi_temp'),
            ('vel_vi_max', 'temp_max', 'vel_vi_max_temp_max'),
            ('vel_vi_min', 'temp_min', 'vel_vi_min_temp_min'),
            ('vel_vi', 'hum_rel', 'vel_vi_hum_rel'),
            ('vel_vi_max', 'hum_rel', 'vel_vi_max_hum_rel'),
            ('dias_lluvia', 'hum_rel', 'dias_lluvia_hum_rel'),
            ('hum_esp', 'temp', 'hum_esp_temp'),
        ]
        for col1, col2, new_col in interactions:
            if col1 in X.columns and col2 in X.columns:
                interaction_data[new_col] = X[col1] * X[col2]
        return interaction_data

    def _create_polynomial_features(self, X):
        important_vars = ['temp', 'temp_max', 'temp_min', 'hum_rel', 'prec', 'vel_vi']
        poly_features = pd.DataFrame(index=X.index)
        for var in important_vars:
            if var in X.columns:
                poly_features[var + '_squared'] = X[var] ** 2
        return poly_features

    def _create_lag_aggregates(self, X):
        lag_aggs = pd.DataFrame(index=X.index)
        for base_var in METEOROLOGICAL_LAG_VARS:
            lag_cols = [col for col in X.columns
                        if col.startswith(base_var + '_lag_')]
            if len(lag_cols) >= 3:
                lag_data = X[lag_cols]
                lag_aggs[base_var + '_lag_mean'] = lag_data.mean(axis=1)
                lag_aggs[base_var + '_lag_std'] = lag_data.std(axis=1)
                lag_aggs[base_var + '_lag_max'] = lag_data.max(axis=1)
                lag_aggs[base_var + '_lag_min'] = lag_data.min(axis=1)
        return lag_aggs

    def _create_trend_features(self, X):
        trend_features = pd.DataFrame(index=X.index)
        trend_vars = ['temp', 'temp_max', 'temp_min', 'hum_rel',
                      'prec', 'dias_lluvia', 'vel_vi']
        for var in trend_vars:
            if var + '_lag_1' in X.columns and var + '_lag_4' in X.columns:
                trend_features[var + '_trend_short'] = (
                    X[var + '_lag_1'] - X[var + '_lag_4'])
            if var + '_lag_4' in X.columns and var + '_lag_8' in X.columns:
                trend_features[var + '_trend_mid'] = (
                    X[var + '_lag_4'] - X[var + '_lag_8'])
            if var + '_lag_8' in X.columns and var + '_lag_12' in X.columns:
                trend_features[var + '_trend_long'] = (
                    X[var + '_lag_8'] - X[var + '_lag_12'])
        return trend_features

    def _create_cumulative_features(self, X):
        cum_features = pd.DataFrame(index=X.index)
        cum_vars = ['prec', 'dias_lluvia', 'temp', 'hum_rel']
        for var in cum_vars:
            for window in [3, 6, 12]:
                cols = [var + '_lag_' + str(i) for i in range(1, window + 1)]
                cols = [c for c in cols if c in X.columns]
                if len(cols) == window:
                    cum_features[var + '_cum_' + str(window) + 'w'] = X[cols].sum(axis=1)
        return cum_features

    def _augment_features(self, X):
        meteo_cols = self._get_meteorological_columns(X)
        X_meteo = X[meteo_cols].copy()
        X_meteo = X_meteo.dropna(axis=1, how='all')

        features_list = [X_meteo]

        interaction_features = self._create_interaction_features(X_meteo)
        if not interaction_features.empty:
            features_list.append(interaction_features)

        poly_features = self._create_polynomial_features(X_meteo)
        if not poly_features.empty:
            features_list.append(poly_features)

        lag_aggs = self._create_lag_aggregates(X_meteo)
        if not lag_aggs.empty:
            features_list.append(lag_aggs)

        trend_features = self._create_trend_features(X_meteo)
        if not trend_features.empty:
            features_list.append(trend_features)

        cum_features = self._create_cumulative_features(X_meteo)
        if not cum_features.empty:
            features_list.append(cum_features)

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
    print("SCRIPT DE DESPLIEGUE - MODELO METEOROLÓGICO DENGUE")
    print("="*60)

    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\6_redes_neuronales\1_red_neuronal_mlp\2_datos\1_raw\3_95_5\modelo_mlp_meteorologico_guardado"
    model = DengueModelDeployment(model_dir)

    print("\nCargando datos de prueba...")
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\6_redes_neuronales\1_red_neuronal_mlp\2_datos\1_raw\3_95_5\2_meteo_epi_2021-2026_1_rezagos_meteo_epi_test_95_5.xlsx"
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
