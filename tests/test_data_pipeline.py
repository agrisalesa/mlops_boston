# tests/test_data_pipeline.py
import numpy as np

def test_processed_no_nans(processed_df):
    assert not processed_df.isna().any().any(), "No deben quedar NaN en el dataset procesado."

def test_processed_types_numeric(processed_df):
    # Todas las columnas salvo la target deben ser numéricas en este caso.
    # Si en tu schema defines target, úsalo; aquí asumimos MEDV
    for col in processed_df.columns:
        assert np.issubdtype(processed_df[col].dtype, np.number), f"La columna {col} debe ser numérica."
