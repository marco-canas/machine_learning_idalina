"""
Script de despliegue del modelo de predicción de casos de dengue
Generado automáticamente - 2026-09-04 11:05:20

Este script permite cargar el modelo entrenado y realizar predicciones
sobre nuevos datos con el mismo formato que los datos de entrenamiento.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler

class DengueModelDeployment:
    """
    Clase para desplegar el modelo de predicción de casos de dengue
    """
    def __init__(self, model_dir):
        """
        Inicializa el modelo cargando los archivos necesarios

        Parameters:
        -----------
        model_dir : str
            Directorio donde se encuentran los archivos del modelo
        """
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        """Carga el modelo y los objetos necesarios desde archivos"""
        # Cargar scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        # Cargar modelo XGBoost
        with open(os.path.join(self.model_dir, 'xgb_model.pkl'), 'rb') as f:
            self.xgb_model = pickle.load(f)

        # Cargar configuración
        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
            self.selected_features = config['selected_features']
            self.X_augmented_columns = config['X_augmented_columns']
            self.numeric_columns = config['numeric_columns']
            self.feature_names = config['feature_names']
            self.best_params = config.get('best_params', {})

        print("✅ Modelo cargado exitosamente")
        print(f"   Características seleccionadas: {len(self.selected_features)}")
        print(f"   Hiperparámetros: {self.best_params}")

    def _create_interaction_features(self, X):
        """Crea características de interacción"""
        interaction_data = pd.DataFrame(index=X.index)

        interactions = [
            ('temp', 'hum_rel', 'temp_hum_rel'),
            ('temp_max', 'hum_rel', 'temp_max_hum_rel'),
            ('prec', 'temp', 'prec_temp'),
            ('dias_lluvia', 'hum_rel', 'dias_lluvia_hum_rel'),
            ('soi', 'sst', 'soi_sst'),
            ('temp', 'soi', 'temp_soi'),
        ]

        for col1, col2, new_col in interactions:
            if col1 in X.columns and col2 in X.columns:
                interaction_data[new_col] = X[col1] * X[col2]

        return interaction_data

    def _create_polynomial_features(self, X):
        """Crea características polinomiales"""
        important_vars = ['temp', 'hum_rel', 'prec']
        poly_features = pd.DataFrame(index=X.index)

        for var in important_vars:
            if var in X.columns:
                poly_features[var + '_squared'] = X[var] ** 2

        return poly_features

    def _create_lag_aggregates(self, X):
        """Crea agregados de rezagos"""
        lag_aggs = pd.DataFrame(index=X.index)

        for base_var in ['temp', 'temp_max', 'temp_min', 'hum_rel', 'prec', 'dias_lluvia']:
            lag_cols = [col for col in X.columns if col.startswith(base_var + '_lag_')]
            if len(lag_cols) >= 3:
                lag_data = X[lag_cols]
                lag_aggs[base_var + '_lag_mean'] = lag_data.mean(axis=1)
                lag_aggs[base_var + '_lag_std'] = lag_data.std(axis=1)

        dengue_lag_cols = [col for col in X.columns if col.startswith('casos_dengue_lag_')]
        if len(dengue_lag_cols) >= 3:
            dengue_lag_data = X[dengue_lag_cols]
            lag_aggs['dengue_lag_mean'] = dengue_lag_data.mean(axis=1)
            lag_aggs['dengue_lag_std'] = dengue_lag_data.std(axis=1)
            lag_aggs['dengue_lag_sum'] = dengue_lag_data.sum(axis=1)

        return lag_aggs

    def _create_rolling_features(self, X):
        """Crea características de tendencia temporal"""
        rolling_features = pd.DataFrame(index=X.index)

        if 'semana_epi' in X.columns:
            rolling_features['week_sin'] = np.sin(2 * np.pi * X['semana_epi'] / 52)
            rolling_features['week_cos'] = np.cos(2 * np.pi * X['semana_epi'] / 52)

        if 'casos_dengue_lag_12' in X.columns:
            if 'casos_dengue_lag_8' in X.columns:
                rolling_features['dengue_trend_4weeks'] = X['casos_dengue_lag_12'] - X['casos_dengue_lag_8']
            if 'casos_dengue_lag_4' in X.columns:
                rolling_features['dengue_trend_8weeks'] = X['casos_dengue_lag_12'] - X['casos_dengue_lag_4']

        return rolling_features

    def _get_numeric_columns(self, X):
        """Identifica columnas numéricas"""
        exclude_cols = ['fecha', 'date', 'datetime', 'timestamp']
        numeric_cols = []

        for col in X.columns:
            if col.lower() in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                numeric_cols.append(col)

        return numeric_cols

    def _augment_features(self, X):
        """Aplica ingeniería de atributos a los datos"""
        numeric_cols = self._get_numeric_columns(X)
        X_numeric = X[numeric_cols]

        features_list = [X_numeric]

        interaction_features = self._create_interaction_features(X_numeric)
        if not interaction_features.empty:
            features_list.append(interaction_features)

        poly_features = self._create_polynomial_features(X_numeric)
        if not poly_features.empty:
            features_list.append(poly_features)

        lag_aggs = self._create_lag_aggregates(X_numeric)
        if not lag_aggs.empty:
            features_list.append(lag_aggs)

        rolling_features = self._create_rolling_features(X_numeric)
        if not rolling_features.empty:
            features_list.append(rolling_features)

        X_augmented = pd.concat(features_list, axis=1)

        X_augmented = X_augmented.replace([np.inf, -np.inf], np.nan)
        X_augmented = X_augmented.fillna(X_augmented.mean())

        return X_augmented

    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos

        Parameters:
        -----------
        X : pandas.DataFrame
            Datos de entrada con las mismas columnas que los datos de entrenamiento

        Returns:
        --------
        numpy.ndarray : Predicciones del modelo
        """
        # Aplicar ingeniería de atributos
        X_augmented = self._augment_features(X)

        # Asegurar que todas las columnas estén presentes
        for col in self.X_augmented_columns:
            if col not in X_augmented.columns:
                X_augmented[col] = 0

        # Reordenar columnas
        X_augmented = X_augmented[self.X_augmented_columns]

        # Escalar
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_augmented),
            columns=X_augmented.columns,
            index=X_augmented.index
        )

        # Seleccionar características
        X_final = X_scaled[self.selected_features]

        # Predecir
        predictions = self.xgb_model.predict(X_final)

        return predictions

    def predict_and_save(self, X, output_path=None):
        """
        Realiza predicciones y las guarda en un archivo

        Parameters:
        -----------
        X : pandas.DataFrame
            Datos de entrada
        output_path : str, optional
            Ruta donde guardar las predicciones
        """
        predictions = self.predict(X)

        # Crear DataFrame con resultados
        results_df = X.copy()
        results_df['prediccion_casos_dengue'] = predictions

        if output_path:
            results_df.to_excel(output_path, index=False)
            print(f"✅ Predicciones guardadas en: {output_path}")

        return results_df

# ===================== EJECUCIÓN DE PRUEBA =====================

if __name__ == "__main__":
    print("="*60)
    print("SCRIPT DE DESPLIEGUE DEL MODELO DE DENGUE")
    print("="*60)

    # Ruta del modelo
    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\3_xgboost\2_datos\1_raw\2_90_10\modelo_guardado_optimizado"

    # Inicializar modelo
    model = DengueModelDeployment(model_dir)

    # Ejemplo de uso con datos de prueba
    print("\nCargando datos de prueba...")
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\3_xgboost\2_datos\1_raw\2_90_10\2_meteo_epi_2021-2026_1_rezagos_meteo_epi_test_90_10.xlsx"
    test_data = pd.read_excel(test_path)

    # Separar variables predictoras
    X_test = test_data.drop('casos_dengue', axis=1)

    # Realizar predicciones
    print("Realizando predicciones...")
    predictions = model.predict(X_test)

    print(f"Predicciones generadas para {len(predictions)} muestras")
    print(f"Estadísticas de predicciones:")
    print(f"  Media: {np.mean(predictions):.2f}")
    print(f"  Mediana: {np.median(predictions):.2f}")
    print(f"  Mínimo: {np.min(predictions):.2f}")
    print(f"  Máximo: {np.max(predictions):.2f}")

    # Guardar predicciones
    output_path = os.path.join(model_dir, "predicciones_ejemplo.xlsx")
    model.predict_and_save(X_test, output_path)
