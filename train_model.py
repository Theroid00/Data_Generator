"""
Model Training Module
Handles training generative models from real CSV datasets using SDV
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Optional, Union, List
import warnings

try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    logging.warning("SDV not available. Using fallback statistical model.")

from logic_rules import DataValidator


class ModelTrainer:
    """Handles training of generative models from CSV data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
    
    def detect_column_types(self, df: pd.DataFrame) -> dict:
        """Automatically detect column types and constraints"""
        column_info = {}
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dtype == 'int64' or all(col_data == col_data.astype(int)):
                    column_info[column] = {
                        'type': 'integer',
                        'min': int(col_data.min()),
                        'max': int(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }
                else:
                    column_info[column] = {
                        'type': 'float',
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                column_info[column] = {
                    'type': 'datetime',
                    'min': col_data.min(),
                    'max': col_data.max()
                }
            else:
                # Categorical or text
                unique_values = col_data.unique()
                if len(unique_values) / len(col_data) < 0.1:  # Low cardinality
                    column_info[column] = {
                        'type': 'categorical',
                        'categories': list(unique_values),
                        'frequencies': col_data.value_counts().to_dict()
                    }
                else:
                    column_info[column] = {
                        'type': 'text',
                        'sample_values': list(unique_values[:10])
                    }
        
        return column_info
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for model training"""
        df_processed = df.copy()
        
        # Convert datetime columns
        for column in df_processed.columns:
            if df_processed[column].dtype == 'object':
                # Try to convert to datetime
                try:
                    df_processed[column] = pd.to_datetime(df_processed[column], errors='ignore')
                except:
                    pass
        
        # Fill missing values with appropriate defaults
        for column in df_processed.columns:
            if df_processed[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[column]):
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(df_processed[column]):
                    df_processed[column].fillna(df_processed[column].mode().iloc[0], inplace=True)
                else:
                    df_processed[column].fillna(df_processed[column].mode().iloc[0], inplace=True)
        
        return df_processed
    
    def train_sdv_model(self, df: pd.DataFrame, model_type: str = 'ctgan') -> Optional[object]:
        """Train SDV model (CTGAN or GaussianCopula)"""
        if not SDV_AVAILABLE:
            return None
        
        try:
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            
            # Initialize synthesizer
            if model_type.lower() == 'ctgan':
                synthesizer = CTGANSynthesizer(metadata, epochs=10, verbose=False)
            else:
                synthesizer = GaussianCopulaSynthesizer(metadata)
            
            # Train the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synthesizer.fit(df)
            
            return synthesizer
            
        except Exception as e:
            self.logger.error(f"Failed to train SDV model: {str(e)}")
            return None
    
    def train_fallback_model(self, df: pd.DataFrame) -> dict:
        """Train a simple statistical model as fallback"""
        model = {
            'type': 'statistical',
            'column_info': self.detect_column_types(df),
            'correlations': {}
        }
        
        # Calculate correlations for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            model['correlations'] = correlation_matrix.to_dict()
        
        return model
    
    def save_model(self, model: Union[object, dict], filepath: Union[str, Path]) -> bool:
        """Save trained model to file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def train_from_csv(self, csv_path: Union[str, Path], model_type: str = 'ctgan') -> Optional[Path]:
        """Train model from CSV file and save it"""
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            self.logger.info(f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Validate data
            if not self.validator.validate_dataframe(df):
                self.logger.warning("Data validation failed, but continuing...")
            
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Try SDV model first, fallback to statistical model
            model = None
            if SDV_AVAILABLE and len(df) >= 100:  # SDV works better with more data
                self.logger.info(f"Training {model_type.upper()} model...")
                model = self.train_sdv_model(df_processed, model_type)
            
            if model is None:
                self.logger.info("Training statistical fallback model...")
                model = self.train_fallback_model(df_processed)
            
            # Save model
            model_filename = f"trained_model_{csv_path.stem}.pkl"
            model_path = Path('output') / model_filename
            
            if self.save_model(model, model_path):
                return model_path
            else:
                return None
        
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return None

    def train_from_multiple_datasets(self, dataset_paths: List[Path], model_type: str = 'ctgan') -> Optional[Path]:
        """
        Train ONE unified model on ALL datasets combined
        Creates a super-powerful model with cross-domain knowledge
        """
        try:
            print(f"\n🧠 Training UNIFIED model on {len(dataset_paths)} datasets...")
            
            # Combine all datasets
            combined_df = pd.DataFrame()
            total_rows = 0
            
            for i, dataset_path in enumerate(dataset_paths, 1):
                try:
                    print(f"   📊 Loading dataset {i}/{len(dataset_paths)}: {dataset_path.name}")
                    df = pd.read_csv(dataset_path)
                    
                    # Add source tracking
                    df['source_dataset'] = dataset_path.stem
                    df['dataset_id'] = i
                    
                    # Combine
                    combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)
                    total_rows += len(df)
                    
                    print(f"     ✅ Added {len(df)} rows")
                    
                except Exception as e:
                    print(f"     ⚠️ Skipped {dataset_path.name}: {e}")
                    continue
            
            if combined_df.empty:
                print("❌ No valid datasets found")
                return None
            
            print(f"\n🔥 COMBINED DATASET: {total_rows} rows from {len(dataset_paths)} sources")
            print(f"📋 Columns: {len(combined_df.columns)} ({list(combined_df.columns[:5])}...)")
            
            # Preprocess combined data
            df_processed = self.preprocess_data(combined_df)
            
            # Train unified model
            model = None
            if SDV_AVAILABLE and len(df_processed) >= 100:
                print(f"🤖 Training unified {model_type.upper()} model...")
                model = self.train_sdv_model(df_processed, model_type)
            
            if model is None:
                print("🔧 Training unified statistical model...")
                model = self.train_fallback_model(df_processed)
            
            if model is None:
                print("❌ Failed to train unified model")
                return None
            
            # Save unified model
            model_filename = f"trained_model_UNIFIED_ALL_DATASETS.pkl"
            model_path = Path('output') / model_filename
            
            if self.save_model(model, model_path):
                print(f"✅ UNIFIED MODEL SAVED: {model_path}")
                print(f"🧠 Model trained on {total_rows} rows across {len(dataset_paths)} domains")
                return model_path
            else:
                print("❌ Failed to save unified model")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}")
            return None
