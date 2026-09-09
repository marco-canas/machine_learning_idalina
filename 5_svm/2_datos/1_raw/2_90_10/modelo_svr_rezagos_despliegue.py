"""
Script de despliegue del modelo SVR para predicción de casos de dengue
Basado SOLO en rezagos de casos de dengue
Generado automáticamente - 2026-09-09 09:34:53
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class DengueModelDeployment:
    """
    Clase para desplegar el modelo SVR de predicción de casos de dengue
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        """Carga el modelo y los objetos necesarios desde archivos"""
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.model_dir, 'svr_model.pkl'), 'rb') as f:
            self.svr_model = pickle.load(f)

        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
            self.selected_features = config['selected_features']
            self.X_augmented_columns = config['X_augmented_columns']
            self.best_params = config.get('best_params', {})

        print("✅ Modelo SVR cargado exitosamente (Solo rezagos de dengue)")
        print(f"   Características seleccionadas: {len(self.selected_features)}")

    def create_lag_aggregates(self, X):
        """Crea agregados de rezagos de dengue"""
        lag_aggs = pd.DataFrame(index=X.index)

        dengue_lag_cols = [col for col in X.columns if col.startswith('casos_dengue_lag_')]

        if len(dengue_lag_cols) >= 3:
            dengue_lag_data = X[dengue_lag_cols]
            lag_aggs['dengue_lag_mean'] = dengue_lag_data.mean(axis=1)
            lag_aggs['dengue_lag_std'] = dengue_lag_data.std(axis=1)
            lag_aggs['dengue_lag_sum'] = dengue_lag_data.sum(axis=1)
            lag_aggs['dengue_lag_max'] = dengue_lag_data.max(axis=1)
            lag_aggs['dengue_lag_min'] = dengue_lag_data.min(axis=1)

        return lag_aggs

    def create_rolling_features(self, X):
        """Crea características de promedio móvil"""
        rolling_features = pd.DataFrame(index=X.index)

        if 'casos_dengue_lag_1' in X.columns and 'casos_dengue_lag_2' in X.columns and 'casos_dengue_lag_3' in X.columns:
            rolling_features['dengue_ma_3w'] = (X['casos_dengue_lag_1'] + X['casos_dengue_lag_2'] + X['casos_dengue_lag_3']) / 3

        return rolling_features

    def create_peak_indicators(self, X):
        """Crea indicadores de picos"""
        peak_features = pd.DataFrame(index=X.index)

        if 'casos_dengue_lag_1' in X.columns:
            peak_features['dengue_level_high'] = (X['casos_dengue_lag_1'] > X['casos_dengue_lag_1'].quantile(0.75)).astype(int)

        return peak_features

    def _augment_features(self, X):
        """Aplica ingeniería de atributos a los datos"""
        X_numeric = X.copy()

        features_list = [X_numeric]

        lag_aggs = self.create_lag_aggregates(X_numeric)
        if not lag_aggs.empty:
            features_list.append(lag_aggs)

        rolling_features = self.create_rolling_features(X_numeric)
        if not rolling_features.empty:
            features_list.append(rolling_features)

        peak_features = self.create_peak_indicators(X_numeric)
        if not peak_features.empty:
            features_list.append(peak_features)

        X_augmented = pd.concat(features_list, axis=1)
        X_augmented = X_augmented.replace([np.inf, -np.inf], np.nan)
        X_augmented = X_augmented.fillna(X_augmented.median())

        return X_augmented

    def predict(self, X):
        """Realiza predicciones sobre nuevos datos"""
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
        predictions = self.svr_model.predict(X_final)

        return predictions

    def predict_and_save(self, X, output_path=None):
        """Realiza predicciones y las guarda en un archivo"""
        predictions = self.predict(X)
        results_df = X.copy()
        results_df['prediccion_casos_dengue'] = predictions

        if output_path:
            results_df.to_excel(output_path, index=False)
            print(f"✅ Predicciones guardadas en: {output_path}")

        return results_df

if __name__ == "__main__":
    print("="*60)
    print("SCRIPT DE DESPLIEGUE DEL MODELO SVR (Solo Rezagos)")
    print("="*60)

    model_dir = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\5_svm\2_datos\1_raw\2_90_10\modelo_svr_rezagos_guardado"
    model = DengueModelDeployment(model_dir)

    print("\nCargando datos de prueba...")
    test_path = r"C:\Users\marco\Documentos\investigacion\machine_learning_idalina\5_svm\2_datos\1_raw\2_90_10\3_meteo_epi_2021-2026_1_rezagos_meteo_test_90_10.xlsx"
    test_data = pd.read_excel(test_path)
    X_test = test_data.drop('casos_dengue', axis=1)

    print("Realizando predicciones...")
    predictions = model.predict(X_test)

    print(f"Predicciones generadas para {len(predictions)} muestras")
    print(f"Estadísticas:")
    print(f"  Media: {np.mean(predictions):.2f}")
    print(f"  Mediana: {np.median(predictions):.2f}")
    print(f"  Mínimo: {np.min(predictions):.2f}")
    print(f"  Máximo: {np.max(predictions):.2f}")

    output_path = os.path.join(model_dir, "predicciones_ejemplo.xlsx")
    model.predict_and_save(X_test, output_path)
