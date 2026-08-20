"""
Integration tests for training/targets.py with real project data.

These tests verify that the targets module works correctly with the actual
data structure used in the project.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from training.targets import (
    add_targets_to_panel,
    validate_no_leakage,
    get_class_weights
)


class TestIntegrationWithRealData:
    """Integration tests using actual project data structure."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Load a sample of real stock data if available, otherwise create mock data."""
        data_path = Path("data/processed/INFY_NS.csv")
        
        if data_path.exists():
            # Use actual data (limited to first 100 rows for speed)
            df = pd.read_csv(data_path, nrows=100)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            # Create mock data matching the expected structure
            dates = pd.date_range('2021-01-01', periods=100, freq='B')
            return pd.DataFrame({
                'Date': dates,
                'Close': 100 + np.cumsum(np.random.randn(100) * 2),
                'High': 105 + np.cumsum(np.random.randn(100) * 2),
                'Low': 95 + np.cumsum(np.random.randn(100) * 2),
                'Open': 100 + np.cumsum(np.random.randn(100) * 2),
                'Volume': np.random.randint(1000000, 10000000, 100),
                'ticker': ['INFY.NS'] * 100,
                'sector': ['IT'] * 100
            })
    
    def test_add_targets_to_real_data(self, sample_stock_data):
        """Test adding targets to real project data structure."""
        df = sample_stock_data.copy()
        
        # Add targets
        panel = add_targets_to_panel(df, horizons=[1, 5, 10])
        
        # Verify all target columns exist
        assert 'return_1d' in panel.columns
        assert 'target_1d' in panel.columns
        assert 'return_5d' in panel.columns
        assert 'target_5d' in panel.columns
        assert 'return_10d' in panel.columns
        assert 'target_10d' in panel.columns
        
        # Verify we have valid targets
        assert panel['return_1d'].notna().sum() > 0
        assert panel['target_1d'].notna().sum() > 0
        
        # Verify no data leakage
        is_valid, msg = validate_no_leakage(panel)
        assert is_valid, f"Data leakage detected: {msg}"
        
        # Verify all targets are in valid range [0, 4]
        valid_targets = panel['target_1d'].dropna()
        assert valid_targets.min() >= 0
        assert valid_targets.max() <= 4
    
    def test_multi_stock_integration(self):
        """Test with multiple stocks (multi-stock panel)."""
        # Create multi-stock data
        dates = pd.date_range('2021-01-01', periods=50, freq='B')
        stocks = ['INFY.NS', 'TCS.NS', 'SBIN.NS']
        
        data_list = []
        for stock in stocks:
            stock_data = pd.DataFrame({
                'Date': dates,
                'Close': 100 + np.cumsum(np.random.randn(50) * 2),
                'ticker': [stock] * 50
            })
            data_list.append(stock_data)
        
        df = pd.concat(data_list, ignore_index=True)
        
        # Sort by ticker and date (required for target generation)
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # Add targets
        panel = add_targets_to_panel(df, horizons=[1, 5])
        
        # Verify each stock has targets
        for stock in stocks:
            stock_panel = panel[panel['ticker'] == stock]
            assert stock_panel['return_1d'].notna().sum() > 0
            assert stock_panel['target_1d'].notna().sum() > 0
            
            # Last row of each stock should have NaN target
            assert pd.isna(stock_panel['target_1d'].iloc[-1])
        
        # Verify no leakage across stocks
        is_valid, msg = validate_no_leakage(panel)
        assert is_valid, f"Data leakage detected in multi-stock panel: {msg}"
    
    def test_class_weights_on_real_distribution(self, sample_stock_data):
        """Test class weight calculation on realistic return distribution."""
        df = sample_stock_data.copy()
        panel = add_targets_to_panel(df, horizons=[1])
        
        # Get class weights
        weights = get_class_weights(panel['target_1d'], method='balanced')
        
        # Verify we have weights for all classes
        assert len(weights) == 5
        assert all(w > 0 for w in weights.values())
        
        # Verify weights are inversely proportional to frequency
        target_counts = panel['target_1d'].value_counts()
        if len(target_counts) >= 2:
            most_common = target_counts.idxmax()
            least_common = target_counts.idxmin()
            
            # Most common class should have lower weight
            assert weights[most_common] < weights[least_common]
    
    def test_edge_case_insufficient_data(self):
        """Test handling of insufficient data (horizon > data length)."""
        # Create very short data
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=3),
            'Close': [100, 102, 101],
            'ticker': ['STOCK'] * 3
        })
        
        # Try to add 5-day targets (longer than data)
        panel = add_targets_to_panel(df, horizons=[5])
        
        # Should complete without error, but all targets will be NaN
        assert 'return_5d' in panel.columns
        assert 'target_5d' in panel.columns
        # All should be NaN since we don't have 5 days of future data
        assert panel['return_5d'].isna().all()
    
    def test_preserves_existing_columns(self, sample_stock_data):
        """Test that adding targets doesn't modify existing columns."""
        df = sample_stock_data.copy()
        original_columns = set(df.columns)
        original_close = df['Close'].copy()
        
        # Add targets
        panel = add_targets_to_panel(df, horizons=[1])
        
        # Verify all original columns are preserved
        for col in original_columns:
            assert col in panel.columns
        
        # Verify Close column wasn't modified
        pd.testing.assert_series_equal(panel['Close'], original_close, check_names=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
