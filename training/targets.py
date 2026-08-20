"""
Target generation module for price prediction system.

This module handles forward return calculation and classification for multi-horizon
price prediction. It implements strict temporal ordering to prevent data leakage.

Key Functions:
    - compute_forward_returns: Calculate forward returns for specified horizons
    - classify_returns: Convert continuous returns to 5-class categorical labels
    - add_targets_to_panel: Add all target variables to feature panel
    - validate_no_leakage: Ensure temporal ordering is preserved
"""

from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Classification thresholds (in decimal form, e.g., 0.03 = 3%)
STRONG_BUY_THRESHOLD = 0.03
BUY_THRESHOLD = 0.01
HOLD_UPPER_THRESHOLD = 0.01
HOLD_LOWER_THRESHOLD = -0.01
SELL_THRESHOLD = -0.03

# Class labels (numeric for model training)
CLASS_STRONG_SELL = 0
CLASS_SELL = 1
CLASS_HOLD = 2
CLASS_BUY = 3
CLASS_STRONG_BUY = 4

# String labels for API
LABEL_STRONG_SELL = "STRONG SELL"
LABEL_SELL = "SELL"
LABEL_HOLD = "HOLD"
LABEL_BUY = "BUY"
LABEL_STRONG_BUY = "STRONG BUY"


def compute_forward_returns(
    df: pd.DataFrame,
    horizon: int = 1,
    price_col: str = "Close",
    date_col: str = "Date",
    group_col: str = "ticker"
) -> pd.Series:
    """
    Compute forward returns for a specified horizon.
    
    Forward return is calculated as: (Close_t+N - Close_t) / Close_t
    where N is the horizon in trading days.
    
    This function handles multi-stock panels by grouping by ticker and ensures
    no data leakage by using explicit date-based indexing.
    
    Args:
        df: DataFrame with price data (must have Date, Close, ticker columns)
        horizon: Number of trading days forward (1, 5, or 10)
        price_col: Name of the price column (default: 'Close')
        date_col: Name of the date column (default: 'Date')
        group_col: Name of the grouping column (default: 'ticker')
    
    Returns:
        pd.Series: Forward returns with the same index as input df
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Date': pd.date_range('2021-01-01', periods=10),
        ...     'Close': [100, 102, 101, 105, 103, 107, 106, 108, 110, 109],
        ...     'ticker': ['STOCK'] * 10
        ... })
        >>> returns = compute_forward_returns(df, horizon=1)
        >>> # returns[0] = (102 - 100) / 100 = 0.02
        >>> # returns[-1] = NaN (no future data available)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_forward_returns")
        return pd.Series(dtype=float, index=df.index)
    
    # Validate required columns
    required_cols = [price_col, date_col, group_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by ticker and date to ensure proper temporal ordering
    df_sorted = df.sort_values([group_col, date_col]).copy()
    
    # Calculate forward returns grouped by ticker
    forward_returns = df_sorted.groupby(group_col)[price_col].transform(
        lambda x: (x.shift(-horizon) - x) / x
    )
    
    # Restore original index order
    forward_returns = forward_returns.reindex(df.index)
    
    logger.info(
        f"Computed forward returns for horizon={horizon}. "
        f"Valid: {forward_returns.notna().sum()}/{len(forward_returns)} "
        f"({forward_returns.notna().mean():.1%})"
    )
    
    return forward_returns


def classify_returns(
    returns: pd.Series,
    return_thresholds: dict = None
) -> pd.Series:
    """
    Classify continuous returns into 5 categorical classes.
    
    Classification bins:
        - Strong Sell (0): return < -3%
        - Sell (1): -3% <= return < -1%
        - Hold (2): -1% <= return <= 1%
        - Buy (3): 1% < return <= 3%
        - Strong Buy (4): return > 3%
    
    Args:
        returns: Series of continuous forward returns (as decimals, e.g., 0.02 = 2%)
        return_thresholds: Optional dict to override default thresholds
            Keys: 'strong_buy', 'buy', 'hold_upper', 'hold_lower', 'sell'
    
    Returns:
        pd.Series: Integer class labels (0-4) with same index as input
        
    Example:
        >>> returns = pd.Series([0.05, 0.02, 0.005, -0.005, -0.02, -0.05])
        >>> classes = classify_returns(returns)
        >>> # [4, 3, 2, 2, 1, 0] = [Strong Buy, Buy, Hold, Hold, Sell, Strong Sell]
    """
    if returns.empty:
        return pd.Series(dtype=int, index=returns.index)
    
    # Use default thresholds or override
    thresholds = {
        'strong_buy': STRONG_BUY_THRESHOLD,
        'buy': BUY_THRESHOLD,
        'hold_upper': HOLD_UPPER_THRESHOLD,
        'hold_lower': HOLD_LOWER_THRESHOLD,
        'sell': SELL_THRESHOLD
    }
    if return_thresholds:
        thresholds.update(return_thresholds)
    
    # Initialize with NaN
    classes = pd.Series(np.nan, index=returns.index, dtype=float)
    
    # Apply classification rules based on requirements:
    # - Strong Buy (4): return > 3%
    # - Buy (3): 1% < return <= 3%
    # - Hold (2): -1% <= return <= 1%
    # - Sell (1): -3% <= return < -1%
    # - Strong Sell (0): return < -3%
    
    # Order matters: apply from most extreme to least extreme to avoid overlaps
    classes[returns > thresholds['strong_buy']] = CLASS_STRONG_BUY
    classes[(returns > thresholds['buy']) & (returns <= thresholds['strong_buy'])] = CLASS_BUY
    classes[(returns >= thresholds['hold_lower']) & (returns <= thresholds['hold_upper'])] = CLASS_HOLD
    classes[(returns >= thresholds['sell']) & (returns < thresholds['hold_lower'])] = CLASS_SELL
    classes[returns < thresholds['sell']] = CLASS_STRONG_SELL
    
    # Convert to integer (NaN will remain NaN)
    classes = classes.astype('Int64')  # Nullable integer type
    
    # Log class distribution
    if classes.notna().sum() > 0:
        class_counts = classes.value_counts().sort_index()
        logger.info(f"Return classification distribution:")
        class_names = {
            CLASS_STRONG_SELL: "Strong Sell",
            CLASS_SELL: "Sell",
            CLASS_HOLD: "Hold",
            CLASS_BUY: "Buy",
            CLASS_STRONG_BUY: "Strong Buy"
        }
        for class_id in range(5):
            count = class_counts.get(class_id, 0)
            pct = count / classes.notna().sum() * 100
            logger.info(f"  {class_names[class_id]} ({class_id}): {count} ({pct:.1f}%)")
    
    return classes


def class_to_label(class_id: int) -> str:
    """
    Convert numeric class ID to string label.
    
    Args:
        class_id: Integer class (0-4)
    
    Returns:
        str: Label string ('STRONG SELL', 'SELL', 'HOLD', 'BUY', 'STRONG BUY')
    """
    label_map = {
        CLASS_STRONG_SELL: LABEL_STRONG_SELL,
        CLASS_SELL: LABEL_SELL,
        CLASS_HOLD: LABEL_HOLD,
        CLASS_BUY: LABEL_BUY,
        CLASS_STRONG_BUY: LABEL_STRONG_BUY
    }
    return label_map.get(class_id, LABEL_HOLD)


def add_targets_to_panel(
    panel: pd.DataFrame,
    horizons: List[int] = [1, 5, 10],
    price_col: str = "Close",
    date_col: str = "Date",
    group_col: str = "ticker"
) -> pd.DataFrame:
    """
    Add forward return targets to feature panel for multiple horizons.
    
    For each horizon, this function adds two columns:
        - return_{horizon}d: Continuous forward return
        - target_{horizon}d: Categorical class label (0-4)
    
    Args:
        panel: Feature panel DataFrame (multi-stock, sorted by ticker and date)
        horizons: List of forward horizons in trading days (default: [1, 5, 10])
        price_col: Name of the price column (default: 'Close')
        date_col: Name of the date column (default: 'Date')
        group_col: Name of the grouping column (default: 'ticker')
    
    Returns:
        pd.DataFrame: Panel with added target columns
        
    Example:
        >>> panel = pd.DataFrame({
        ...     'Date': pd.date_range('2021-01-01', periods=20),
        ...     'Close': np.random.uniform(100, 110, 20),
        ...     'ticker': ['STOCK'] * 20,
        ...     'feature1': np.random.randn(20)
        ... })
        >>> panel_with_targets = add_targets_to_panel(panel, horizons=[1, 5])
        >>> # Adds: return_1d, target_1d, return_5d, target_5d
    """
    if panel.empty:
        logger.warning("Empty panel provided to add_targets_to_panel")
        return panel
    
    logger.info(f"Adding targets for horizons: {horizons}")
    
    # Create a copy to avoid modifying the original
    panel_with_targets = panel.copy()
    
    for horizon in horizons:
        logger.info(f"Processing horizon: {horizon} days")
        
        # Compute forward returns
        return_col = f"return_{horizon}d"
        panel_with_targets[return_col] = compute_forward_returns(
            df=panel_with_targets,
            horizon=horizon,
            price_col=price_col,
            date_col=date_col,
            group_col=group_col
        )
        
        # Classify returns into categories
        target_col = f"target_{horizon}d"
        panel_with_targets[target_col] = classify_returns(
            panel_with_targets[return_col]
        )
        
        # Log statistics
        valid_returns = panel_with_targets[return_col].notna().sum()
        valid_targets = panel_with_targets[target_col].notna().sum()
        logger.info(
            f"  Added {return_col}: {valid_returns} valid values "
            f"(mean={panel_with_targets[return_col].mean():.4f}, "
            f"std={panel_with_targets[return_col].std():.4f})"
        )
        logger.info(f"  Added {target_col}: {valid_targets} valid values")
    
    return panel_with_targets


def validate_no_leakage(
    df: pd.DataFrame,
    date_col: str = "Date",
    group_col: str = "ticker",
    target_cols: List[str] = None
) -> Tuple[bool, str]:
    """
    Validate that forward returns do not leak future information into past.
    
    This function checks:
        1. Data is sorted by ticker and date
        2. Target values are only present for dates with sufficient future data
        3. No target values appear in the last N rows of each group (N = horizon)
    
    Args:
        df: DataFrame with targets
        date_col: Name of the date column
        group_col: Name of the grouping column
        target_cols: List of target column names to validate (if None, auto-detect)
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if no leakage detected
            - error_message: Empty string if valid, otherwise describes the issue
            
    Example:
        >>> panel_with_targets = add_targets_to_panel(panel, horizons=[1, 5])
        >>> is_valid, msg = validate_no_leakage(panel_with_targets)
        >>> assert is_valid, f"Data leakage detected: {msg}"
    """
    if df.empty:
        return True, ""
    
    # Auto-detect target columns if not specified
    if target_cols is None:
        target_cols = [col for col in df.columns if col.startswith('target_')]
    
    if not target_cols:
        logger.warning("No target columns found for validation")
        return True, ""
    
    logger.info(f"Validating temporal ordering for columns: {target_cols}")
    
    # Check 1: Verify data is sorted by group and date
    df_sorted = df.sort_values([group_col, date_col])
    if not df_sorted.index.equals(df.index):
        return False, f"DataFrame is not sorted by [{group_col}, {date_col}]"
    
    # Check 2: Verify targets are NaN at the end of each group
    for target_col in target_cols:
        # Extract horizon from column name (e.g., 'target_5d' -> 5)
        try:
            horizon = int(target_col.split('_')[1].replace('d', ''))
        except (IndexError, ValueError):
            logger.warning(f"Could not extract horizon from {target_col}, skipping")
            continue
        
        # Check each group
        for ticker in df[group_col].unique():
            group_df = df[df[group_col] == ticker]
            
            # Last 'horizon' rows should have NaN targets (no future data available)
            last_n_targets = group_df[target_col].iloc[-horizon:]
            
            if last_n_targets.notna().any():
                non_nan_count = last_n_targets.notna().sum()
                return False, (
                    f"Data leakage detected for {ticker}: "
                    f"{target_col} has {non_nan_count} non-NaN values in last {horizon} rows. "
                    f"This indicates future data may be leaking into past predictions."
                )
    
    logger.info("Temporal ordering validation passed: no data leakage detected")
    return True, ""


def get_class_weights(
    targets: pd.Series,
    method: str = "balanced"
) -> dict:
    """
    Compute class weights for handling imbalanced target distributions.
    
    Args:
        targets: Series of target class labels (0-4)
        method: Weighting method ('balanced' or 'sqrt')
            - 'balanced': weight = n_samples / (n_classes * class_count)
            - 'sqrt': weight = sqrt(total_count / class_count)
    
    Returns:
        dict: Mapping of class_id -> weight
        
    Example:
        >>> targets = pd.Series([0, 1, 2, 2, 2, 3, 4])
        >>> weights = get_class_weights(targets)
        >>> # Class 2 (Hold) has weight < 1 since it's overrepresented
        >>> # Classes 0, 1, 3, 4 have weight > 1 since they're underrepresented
    """
    # Remove NaN values
    valid_targets = targets.dropna()
    
    if valid_targets.empty:
        logger.warning("No valid targets for class weight calculation")
        return {i: 1.0 for i in range(5)}
    
    # Count occurrences of each class
    class_counts = valid_targets.value_counts().to_dict()
    n_samples = len(valid_targets)
    n_classes = 5  # Always 5 classes
    
    # Ensure all classes are present (even if count is 0)
    for class_id in range(5):
        if class_id not in class_counts:
            class_counts[class_id] = 0
    
    # Compute weights based on method
    if method == "balanced":
        # Scikit-learn style: n_samples / (n_classes * class_count)
        weights = {
            class_id: n_samples / (n_classes * max(count, 1))
            for class_id, count in class_counts.items()
        }
    elif method == "sqrt":
        # Square root scaling (less aggressive than balanced)
        weights = {
            class_id: np.sqrt(n_samples / max(count, 1))
            for class_id, count in class_counts.items()
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'sqrt'")
    
    # Normalize weights to have mean of 1.0
    weight_mean = np.mean(list(weights.values()))
    weights = {class_id: w / weight_mean for class_id, w in weights.items()}
    
    logger.info(f"Class weights (method={method}):")
    for class_id in range(5):
        logger.info(f"  Class {class_id}: {weights[class_id]:.3f} (count: {class_counts[class_id]})")
    
    return weights


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=100, freq='B')  # Business days
    
    sample_data = pd.DataFrame({
        'Date': np.tile(dates, 2),
        'ticker': ['STOCK_A'] * 100 + ['STOCK_B'] * 100,
        'Close': np.concatenate([
            100 + np.cumsum(np.random.randn(100) * 2),
            200 + np.cumsum(np.random.randn(100) * 3)
        ])
    })
    
    print("Sample data created:")
    print(sample_data.head(10))
    print()
    
    # Add targets
    print("Adding targets for horizons [1, 5, 10]...")
    panel_with_targets = add_targets_to_panel(sample_data, horizons=[1, 5, 10])
    print()
    
    print("Panel with targets (first 10 rows):")
    print(panel_with_targets.head(10))
    print()
    
    # Validate no leakage
    print("Validating temporal ordering...")
    is_valid, msg = validate_no_leakage(panel_with_targets)
    if is_valid:
        print("✓ No data leakage detected")
    else:
        print(f"✗ Data leakage detected: {msg}")
    print()
    
    # Show class distribution for 1-day target
    print("Class distribution for 1-day target:")
    print(panel_with_targets['target_1d'].value_counts().sort_index())
    print()
    
    # Show class weights
    print("Computing class weights...")
    weights = get_class_weights(panel_with_targets['target_1d'])
    print()
    
    print("Example complete!")
