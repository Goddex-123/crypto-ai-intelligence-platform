"""
Advanced Machine Learning Models for Cryptocurrency Price Prediction
Includes LSTM, Transformer, XGBoost, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, 
    Input, MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D, Embedding, Concatenate
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Traditional ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Utilities
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import os
from datetime import datetime


class SequencePreparator:
    """
    Prepare sequences for time series models
    """
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def create_sequences(self, data: np.ndarray, 
                        target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Target (future value)
            if target is not None:
                target_val = target[i + self.sequence_length + self.prediction_horizon - 1]
                targets.append(target_val)
        
        return np.array(sequences), np.array(targets)


class AdvancedLSTM:
    """
    Advanced LSTM model with attention and regularization
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def _get_default_config(self) -> Dict:
        return {
            'sequence_length': 60,
            'lstm_units': [128, 64, 32],
            'dense_units': [64, 32],
            'dropout_rate': 0.3,
            'l2_reg': 0.001,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'patience': 20
        }
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build advanced LSTM model
        """
        inputs = Input(shape=input_shape)
        x = inputs
        
        # LSTM layers with dropout and regularization
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = i < len(self.config['lstm_units']) - 1
            
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate'],
                kernel_regularizer=l2(self.config['l2_reg']),
                name=f'lstm_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Dense layers
        for i, units in enumerate(self.config['dense_units']):
            x = Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(self.config['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            x = Dropout(self.config['dropout_rate'], name=f'dropout_{i+1}')(x)
            x = BatchNormalization(name=f'bn_dense_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='advanced_lstm')
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the LSTM model
        """
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)


class TransformerModel:
    """
    Transformer model for time series prediction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.history = None
        
    def _get_default_config(self) -> Dict:
        return {
            'sequence_length': 60,
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 512,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'patience': 20
        }
    
    def transformer_encoder(self, inputs: tf.Tensor, 
                           d_model: int, num_heads: int, 
                           dff: int, dropout_rate: float) -> tf.Tensor:
        """
        Transformer encoder layer
        """
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads
        )(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        
        # Add & Norm
        x1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed Forward Network
        ffn_output = Dense(dff, activation='relu')(x1)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        # Add & Norm
        x2 = LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
        
        return x2
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build Transformer model
        """
        inputs = Input(shape=input_shape)
        
        # Input projection to d_model dimensions
        x = Dense(self.config['d_model'])(inputs)
        
        # Positional encoding (simple version)
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_encoding = self.positional_encoding(input_shape[0], self.config['d_model'])
        x += pos_encoding[:input_shape[0], :]
        
        x = Dropout(self.config['dropout_rate'])(x)
        
        # Transformer encoder layers
        for _ in range(self.config['num_layers']):
            x = self.transformer_encoder(
                x,
                self.config['d_model'],
                self.config['num_heads'], 
                self.config['dff'],
                self.config['dropout_rate']
            )
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Final dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config['dropout_rate'])(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config['dropout_rate'])(x)
        
        # Output
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='transformer')
        
        # Compile
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """
        Create positional encoding
        """
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices  
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """
        Calculate angles for positional encoding
        """
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the Transformer model
        """
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)


class EnsembleModel:
    """
    Ensemble model combining multiple ML algorithms
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.model_weights = {}
        self.scalers = {}
        self.is_trained = False
        
    def _get_default_config(self) -> Dict:
        return {
            'models': ['xgboost', 'lightgbm', 'catboost', 'random_forest'],
            'ensemble_method': 'weighted_average',  # 'simple_average', 'stacking'
            'cv_folds': 5,
            'random_state': 42
        }
    
    def _initialize_models(self) -> Dict:
        """
        Initialize all base models
        """
        models = {}
        
        if 'xgboost' in self.config['models']:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        if 'lightgbm' in self.config['models']:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config['random_state'],
                n_jobs=-1,
                verbose=-1
            )
        
        if 'catboost' in self.config['models']:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                verbose=False
            )
        
        if 'random_forest' in self.config['models']:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                random_state=self.config['random_state'],
                n_jobs=-1
            )
        
        if 'gradient_boosting' in self.config['models']:
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config['random_state']
            )
        
        return models
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train all models in the ensemble
        """
        # Initialize models
        self.models = self._initialize_models()
        
        # Convert to DataFrame if numpy array
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_val, np.ndarray) and X_val is not None:
            X_val = pd.DataFrame(X_val)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        model_scores = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Train model on fold
                if name in ['xgboost', 'lightgbm']:
                    model.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_fold_train, y_fold_train)
                
                # Predict and score
                y_pred = model.predict(X_fold_val)
                score = mean_squared_error(y_fold_val, y_pred)
                cv_scores.append(score)
            
            # Store average CV score
            avg_score = np.mean(cv_scores)
            model_scores[name] = avg_score
            print(f"{name} - Average CV MSE: {avg_score:.6f}")
            
            # Train on full training set
            if name in ['xgboost', 'lightgbm'] and X_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
        
        # Calculate model weights (inverse of MSE)
        if self.config['ensemble_method'] == 'weighted_average':
            total_inverse_score = sum(1/score for score in model_scores.values())
            self.model_weights = {
                name: (1/score) / total_inverse_score 
                for name, score in model_scores.items()
            }
        else:
            # Equal weights
            num_models = len(model_scores)
            self.model_weights = {name: 1/num_models for name in model_scores.keys()}
        
        print(f"Model weights: {self.model_weights}")
        
        self.is_trained = True
        return model_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained yet")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Ensemble predictions
        if self.config['ensemble_method'] == 'simple_average':
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        elif self.config['ensemble_method'] == 'weighted_average':
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                ensemble_pred += self.model_weights[name] * pred
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, Dict]:
        """
        Get feature importance from all models
        """
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):  # CatBoost
                importance_dict[name] = model.get_feature_importance()
        
        return importance_dict


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (handling division by zero)
        mask = y_true != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf
        
        # Directional accuracy
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        metrics['directional_accuracy'] = directional_accuracy
        
        # Theil's U statistic
        if len(y_true) > 1:
            naive_pred = np.roll(y_true, 1)[1:]  # Naive prediction (previous value)
            y_true_subset = y_true[1:]
            y_pred_subset = y_pred[1:]
            
            mse_model = mean_squared_error(y_true_subset, y_pred_subset)
            mse_naive = mean_squared_error(y_true_subset, naive_pred)
            
            if mse_naive > 0:
                metrics['theil_u'] = np.sqrt(mse_model) / np.sqrt(mse_naive)
            else:
                metrics['theil_u'] = np.inf
        
        return metrics
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual"):
        """
        Plot predictions vs actual values
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 8))
            
            # Time series plot
            plt.subplot(2, 2, 1)
            plt.plot(y_true, label='Actual', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7)
            plt.title(f'{title} - Time Series')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Scatter plot
            plt.subplot(2, 2, 2)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Predicted vs Actual')
            plt.grid(True)
            
            # Residuals
            plt.subplot(2, 2, 3)
            residuals = y_pred - y_true
            plt.plot(residuals)
            plt.title('Residuals')
            plt.xlabel('Time')
            plt.ylabel('Residual')
            plt.grid(True)
            
            # Residuals histogram
            plt.subplot(2, 2, 4)
            plt.hist(residuals, bins=50, alpha=0.7)
            plt.title('Residuals Distribution')
            plt.xlabel('Residual')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    sequence_length = 60
    
    # Create synthetic data with trend and seasonality
    t = np.linspace(0, 10*np.pi, n_samples)
    trend = np.linspace(50000, 60000, n_samples)
    seasonal = 5000 * np.sin(t) + 2000 * np.cos(2*t)
    noise = np.random.normal(0, 1000, n_samples)
    
    price_data = trend + seasonal + noise
    
    # Create feature matrix
    feature_data = np.random.randn(n_samples, n_features)
    
    # Add price as a feature
    full_data = np.column_stack([price_data.reshape(-1, 1), feature_data])
    
    # Prepare sequences
    seq_prep = SequencePreparator(sequence_length=sequence_length)
    X, y = seq_prep.create_sequences(full_data, price_data)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Validation split
    val_split_idx = int(0.8 * len(X_train))
    X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    
    # Test LSTM model
    print("\n=== Testing LSTM Model ===")
    lstm_model = AdvancedLSTM()
    lstm_history = lstm_model.train(X_train, y_train, X_val, y_val)
    lstm_pred = lstm_model.predict(X_test)
    lstm_metrics = ModelEvaluator.evaluate_model(y_test, lstm_pred.flatten())
    
    print("LSTM Metrics:")
    for metric, value in lstm_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Test ensemble model (using flattened features for traditional ML)
    print("\n=== Testing Ensemble Model ===")
    # Flatten sequences for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    ensemble_model = EnsembleModel()
    ensemble_scores = ensemble_model.train(X_train_flat, y_train, X_val_flat, y_val)
    ensemble_pred = ensemble_model.predict(X_test_flat)
    ensemble_metrics = ModelEvaluator.evaluate_model(y_test, ensemble_pred)
    
    print("Ensemble Metrics:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nModel training and evaluation completed successfully!")
