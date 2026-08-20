"""
Unit tests for training/targets.py module.

Tests cover:
    - Forward return calculation accuracy
    - Return classification correctness
    - Multi-stock panel handling
    - Edge cases (missing data, single stock, etc.)
    - Temporal ordering validation
"""

import pytest
import pandas as pd
import numpy as np
from training.targets import (
    compute_forward_returns,
    classify_returns,
    add_targets_to_panel,
    validate_no_leakage,
    get_class_weights,
    class_to_label,
    CLASS_STRONG_SELL,
    CLASS_SELL,
    CLASS_HOLD,
    CLASS_BUY,
    CLASS_STRONG_BUY,
    LABEL_STRONG_SELL,
    LABEL_SELL,
    LABEL_HOLD,
    LABEL_BUY,
    LABEL_STRONG_BUY
)


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""
    
    def test_basic_1day_return(self):
        """Test basic 1-day forward return calculation."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=5),
            'Close': [100, 102, 101, 105, 103],
            'ticker': ['STOCK'] * 5
        })
        
        returns = compute_forward_returns(df, horizon=1)
        
        # Expected: (102-100)/100=0.02, (101-102)/102=-0.0098, (105-101)/101=0.0396, (103-105)/105=-0.0190, NaN
        assert abs(returns.iloc[0] - 0.02) < 1e-6
        assert abs(returns.iloc[1] - ((101-102)/102)) < 1e-6
        assert abs(returns.iloc[2] - ((105-101)/101)) < 1e-6
        assert abs(returns.iloc[3] - ((103-105)/105)) < 1e-6
        assert pd.isna(returns.iloc[4])  # Last value should be NaN
    
    def test_5day_return(self):
        """Test 5-day forward return calculation."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=10),
            'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            'ticker': ['STOCK'] * 10
        })
        
        returns = compute_forward_returns(df, horizon=5)
        
        # Expected: (110-100)/100=0.10, (112-102)/102=0.098, etc.
        assert abs(returns.iloc[0] - 0.10) < 1e-6
        assert abs(returns.iloc[1] - ((112-102)/102)) < 1e-6
        # Last 5 should be NaN
        assert returns.iloc[-5:].isna().all()
    
    def test_multi_stock_panel(self):
        """Test forward return calculation on multi-stock panel."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=5).tolist() * 2,
            'Close': [100, 102, 101, 105, 103, 200, 204, 202, 210, 206],
            'ticker': ['STOCK_A'] * 5 + ['STOCK_B'] * 5
        })
        
        returns = compute_forward_returns(df, horizon=1)
        
        # Check STOCK_A
        assert abs(returns.iloc[0] - 0.02) < 1e-6
        # Check STOCK_B
        assert abs(returns.iloc[5] - 0.02) < 1e-6
        # Last value of each group should be NaN
        assert pd.isna(returns.iloc[4])
        assert pd.isna(returns.iloc[9])
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'Date': [], 'Close': [], 'ticker': []})
        returns = compute_forward_returns(df, horizon=1)
        assert returns.empty
    
    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2021-01-01')],
            'Close': [100],
            'ticker': ['STOCK']
        })
        
        returns = compute_forward_returns(df, horizon=1)
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])
    
    def test_missing_columns(self):
        """Test error handling when required columns are missing."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=5),
            'Close': [100, 102, 101, 105, 103]
            # Missing 'ticker' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_forward_returns(df, horizon=1)


class TestClassifyReturns:
    """Tests for classify_returns function."""
    
    def test_classification_thresholds(self):
        """Test that classification bins match requirements exactly."""
        # Based on design doc:
        # Strong Buy (4): return > 3%
        # Buy (3): 1% < return <= 3%
        # Hold (2): -1% <= return <= 1%
        # Sell (1): -3% <= return < -1%
        # Strong Sell (0): return < -3%
        returns = pd.Series([
            0.05,   # > 3% -> Strong Buy (4)
            0.03,   # = 3% -> Buy (3) [boundary: 1% < return <= 3%]
            0.025,  # 1-3% -> Buy (3)
            0.01,   # = 1% -> Hold (2) [boundary: -1% <= return <= 1%]
            0.005,  # -1 to 1% -> Hold (2)
            0.0,    # 0% -> Hold (2)
            -0.005, # -1 to 1% -> Hold (2)
            -0.01,  # = -1% -> Hold (2) [boundary: -1% <= return <= 1%]
            -0.015, # -3 to -1% -> Sell (1)
            -0.03,  # = -3% -> Sell (1) [boundary: -3% <= return < -1%]
            -0.05   # < -3% -> Strong Sell (0)
        ])
        
        classes = classify_returns(returns)
        
        expected = [
            CLASS_STRONG_BUY,
            CLASS_BUY,        # 3% is Buy (upper boundary)
            CLASS_BUY,
            CLASS_HOLD,       # 1% is Hold (upper boundary)
            CLASS_HOLD,
            CLASS_HOLD,
            CLASS_HOLD,
            CLASS_HOLD,
            CLASS_SELL,
            CLASS_SELL,
            CLASS_STRONG_SELL
        ]
        
        np.testing.assert_array_equal(classes, expected)
    
    def test_boundary_values(self):
        """Test exact boundary values for each class."""
        # Test exact thresholds based on design doc
        # Buy (3): 1% < return <= 3%
        # Hold (2): -1% <= return <= 1%
        # Sell (1): -3% <= return < -1%
        returns = pd.Series([
            0.03,   # = 3% -> Buy (upper boundary of Buy)
            0.01,   # = 1% -> Hold (upper boundary of Hold)
            -0.01,  # = -1% -> Hold (lower boundary of Hold)
            -0.03   # = -3% -> Sell (lower boundary of Sell)
        ])
        
        classes = classify_returns(returns)
        
        expected = [
            CLASS_BUY,   # 3% is Buy, not Strong Buy
            CLASS_HOLD,  # 1% is Hold, not Buy
            CLASS_HOLD,
            CLASS_SELL
        ]
        
        np.testing.assert_array_equal(classes, expected)
    
    def test_with_nan_values(self):
        """Test handling of NaN values in returns."""
        returns = pd.Series([0.02, np.nan, -0.02, np.nan, 0.0])
        
        classes = classify_returns(returns)
        
        assert classes.iloc[0] == CLASS_BUY
        assert pd.isna(classes.iloc[1])
        assert classes.iloc[2] == CLASS_SELL
        assert pd.isna(classes.iloc[3])
        assert classes.iloc[4] == CLASS_HOLD
    
    def test_empty_series(self):
        """Test handling of empty Series."""
        returns = pd.Series(dtype=float)
        classes = classify_returns(returns)
        assert classes.empty


class TestClassToLabel:
    """Tests for class_to_label function."""
    
    def test_all_class_labels(self):
        """Test conversion of all class IDs to labels."""
        assert class_to_label(CLASS_STRONG_SELL) == LABEL_STRONG_SELL
        assert class_to_label(CLASS_SELL) == LABEL_SELL
        assert class_to_label(CLASS_HOLD) == LABEL_HOLD
        assert class_to_label(CLASS_BUY) == LABEL_BUY
        assert class_to_label(CLASS_STRONG_BUY) == LABEL_STRONG_BUY
    
    def test_invalid_class_id(self):
        """Test handling of invalid class ID."""
        # Should return HOLD as default
        assert class_to_label(99) == LABEL_HOLD
        assert class_to_label(-1) == LABEL_HOLD


class TestAddTargetsToPanel:
    """Tests for add_targets_to_panel function."""
    
    def test_single_horizon(self):
        """Test adding targets for a single horizon."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=10),
            'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            'ticker': ['STOCK'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1])
        
        assert 'return_1d' in panel.columns
        assert 'target_1d' in panel.columns
        assert panel['return_1d'].notna().sum() == 9  # All except last
        assert panel['target_1d'].notna().sum() == 9
    
    def test_multiple_horizons(self):
        """Test adding targets for multiple horizons."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=20),
            'Close': np.random.uniform(100, 110, 20),
            'ticker': ['STOCK'] * 20
        })
        
        panel = add_targets_to_panel(df, horizons=[1, 5, 10])
        
        # Check all columns exist
        assert 'return_1d' in panel.columns
        assert 'target_1d' in panel.columns
        assert 'return_5d' in panel.columns
        assert 'target_5d' in panel.columns
        assert 'return_10d' in panel.columns
        assert 'target_10d' in panel.columns
        
        # Check valid counts
        assert panel['return_1d'].notna().sum() == 19
        assert panel['return_5d'].notna().sum() == 15
        assert panel['return_10d'].notna().sum() == 10
    
    def test_multi_stock_targets(self):
        """Test adding targets to multi-stock panel."""
        dates = pd.date_range('2021-01-01', periods=10)
        df = pd.DataFrame({
            'Date': dates.tolist() * 2,
            'Close': list(range(100, 110)) + list(range(200, 210)),
            'ticker': ['STOCK_A'] * 10 + ['STOCK_B'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1])
        
        # Check that each stock group has correct number of valid targets
        stock_a = panel[panel['ticker'] == 'STOCK_A']
        stock_b = panel[panel['ticker'] == 'STOCK_B']
        
        assert stock_a['return_1d'].notna().sum() == 9
        assert stock_b['return_1d'].notna().sum() == 9
    
    def test_empty_panel(self):
        """Test handling of empty panel."""
        df = pd.DataFrame({'Date': [], 'Close': [], 'ticker': []})
        panel = add_targets_to_panel(df, horizons=[1])
        assert panel.empty


class TestValidateNoLeakage:
    """Tests for validate_no_leakage function."""
    
    def test_valid_no_leakage(self):
        """Test validation passes when no leakage is present."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=10),
            'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            'ticker': ['STOCK'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1, 5])
        is_valid, msg = validate_no_leakage(panel, target_cols=['target_1d', 'target_5d'])
        
        assert is_valid
        assert msg == ""
    
    def test_multi_stock_validation(self):
        """Test validation on multi-stock panel."""
        dates = pd.date_range('2021-01-01', periods=10)
        df = pd.DataFrame({
            'Date': dates.tolist() * 2,
            'Close': list(range(100, 110)) + list(range(200, 210)),
            'ticker': ['STOCK_A'] * 10 + ['STOCK_B'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1, 5])
        is_valid, msg = validate_no_leakage(panel)
        
        assert is_valid
        assert msg == ""
    
    def test_detects_leakage(self):
        """Test that validation detects data leakage."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=10),
            'Close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
            'ticker': ['STOCK'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1])
        
        # Introduce leakage by filling the last target value
        panel.loc[panel.index[-1], 'target_1d'] = CLASS_BUY
        
        is_valid, msg = validate_no_leakage(panel, target_cols=['target_1d'])
        
        assert not is_valid
        assert "Data leakage detected" in msg
    
    def test_empty_dataframe(self):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame({'Date': [], 'Close': [], 'ticker': []})
        is_valid, msg = validate_no_leakage(df)
        
        assert is_valid
        assert msg == ""


class TestGetClassWeights:
    """Tests for get_class_weights function."""
    
    def test_balanced_weights(self):
        """Test balanced class weights calculation."""
        targets = pd.Series([
            CLASS_HOLD, CLASS_HOLD, CLASS_HOLD, CLASS_HOLD,  # 4 Hold
            CLASS_BUY, CLASS_BUY,  # 2 Buy
            CLASS_SELL  # 1 Sell
        ])
        
        weights = get_class_weights(targets, method='balanced')
        
        # Hold should have lower weight (overrepresented)
        # Sell should have higher weight (underrepresented)
        assert weights[CLASS_HOLD] < weights[CLASS_SELL]
        assert weights[CLASS_HOLD] < weights[CLASS_BUY]
    
    def test_sqrt_weights(self):
        """Test sqrt class weights calculation."""
        targets = pd.Series([
            CLASS_HOLD, CLASS_HOLD, CLASS_HOLD, CLASS_HOLD,
            CLASS_BUY, CLASS_BUY,
            CLASS_SELL
        ])
        
        weights = get_class_weights(targets, method='sqrt')
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
        # Should have 5 classes
        assert len(weights) == 5
    
    def test_all_classes_present(self):
        """Test that weights are computed for all 5 classes even if some are missing."""
        targets = pd.Series([CLASS_HOLD, CLASS_BUY])  # Only 2 classes
        
        weights = get_class_weights(targets, method='balanced')
        
        # Should have weights for all 5 classes
        assert len(weights) == 5
        assert all(class_id in weights for class_id in range(5))
    
    def test_empty_targets(self):
        """Test handling of empty targets."""
        targets = pd.Series(dtype=int)
        weights = get_class_weights(targets)
        
        # Should return equal weights for all classes
        assert len(weights) == 5
        assert all(w == 1.0 for w in weights.values())


class TestEdgeCases:
    """Tests for various edge cases and error conditions."""
    
    def test_unsorted_data(self):
        """Test that unsorted data is handled correctly."""
        # Create intentionally unsorted data
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2021-01-05'), pd.Timestamp('2021-01-01'),
                     pd.Timestamp('2021-01-03'), pd.Timestamp('2021-01-02'),
                     pd.Timestamp('2021-01-04')],
            'Close': [105, 100, 102, 101, 104],
            'ticker': ['STOCK'] * 5
        })
        
        # Should still compute correct returns after internal sorting
        returns = compute_forward_returns(df, horizon=1)
        
        # The function should handle sorting internally
        assert returns.notna().sum() == 4  # All except last chronological
    
    def test_very_large_horizon(self):
        """Test handling of horizon larger than available data."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=5),
            'Close': [100, 102, 104, 106, 108],
            'ticker': ['STOCK'] * 5
        })
        
        returns = compute_forward_returns(df, horizon=10)
        
        # All returns should be NaN since horizon > data length
        assert returns.isna().all()
    
    def test_constant_prices(self):
        """Test handling of constant prices (no returns)."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=10),
            'Close': [100] * 10,
            'ticker': ['STOCK'] * 10
        })
        
        panel = add_targets_to_panel(df, horizons=[1])
        
        # All returns should be 0, classified as HOLD
        assert (panel['return_1d'].dropna() == 0).all()
        assert (panel['target_1d'].dropna() == CLASS_HOLD).all()
    
    def test_negative_prices(self):
        """Test handling of negative prices (edge case, though unrealistic)."""
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=5),
            'Close': [-100, -102, -104, -106, -108],
            'ticker': ['STOCK'] * 5
        })
        
        # Should still compute returns correctly
        returns = compute_forward_returns(df, horizon=1)
        
        # Return from -100 to -102 is (-102 - (-100)) / (-100) = -2 / -100 = 0.02
        assert abs(returns.iloc[0] - 0.02) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
