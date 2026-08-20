"""
Core interfaces for the reasoning and explanation engine.

This module defines interfaces for model explanation components including
SHAP, attention weights, and gradient attribution methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd


@dataclass
class ExplanationOutput:
    """
    Standardized output structure for model explanations.
    
    Attributes:
        feature_attributions: Dictionary mapping feature names to their attribution scores
        explanation_type: Type of explanation (e.g., 'shap', 'attention', 'gradient')
        base_value: Baseline/expected value for the prediction
        instance_data: The input instance being explained
        metadata: Additional explanation-specific information
    """
    feature_attributions: Dict[str, float]
    explanation_type: str
    base_value: Optional[float] = None
    instance_data: Optional[Union[np.ndarray, pd.Series]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate explanation output after initialization."""
        if not isinstance(self.feature_attributions, dict):
            raise TypeError("feature_attributions must be a dictionary")
        
        if not isinstance(self.explanation_type, str):
            raise TypeError("explanation_type must be a string")
        
        valid_types = ['shap', 'attention', 'gradient', 'integrated_gradient', 'lime']
        if self.explanation_type not in valid_types:
            raise ValueError(f"explanation_type must be one of {valid_types}")


class ExplainerInterface(ABC):
    """
    Abstract interface for model explanation components.
    
    This ensures consistency across SHAP, attention-based, and gradient-based
    explanation methods in the reasoning engine.
    """
    
    @abstractmethod
    def explain(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        instance_idx: Optional[int] = None,
        **kwargs
    ) -> Union[ExplanationOutput, List[ExplanationOutput]]:
        """
        Generate explanations for model predictions.
        
        Args:
            model: The trained model to explain
            X: Input features to explain
            instance_idx: Optional specific instance index to explain (if None, explain all)
            **kwargs: Explainer-specific parameters
            
        Returns:
            ExplanationOutput for single instance or list for multiple instances
        """
        pass
    
    @abstractmethod
    def explain_global(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate global feature importance across all instances.
        
        Args:
            model: The trained model to explain
            X: Representative sample of input features
            **kwargs: Explainer-specific parameters
            
        Returns:
            Dictionary mapping feature names to global importance scores
        """
        pass
    
    @abstractmethod
    def visualize(
        self,
        explanation: ExplanationOutput,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create visualization of the explanation.
        
        Args:
            explanation: ExplanationOutput object to visualize
            output_path: Optional path to save visualization
            **kwargs: Visualization-specific parameters
            
        Returns:
            Visualization object (e.g., matplotlib figure)
        """
        pass
    
    @property
    @abstractmethod
    def explainer_name(self) -> str:
        """Return the name/identifier of this explainer."""
        pass
    
    @property
    @abstractmethod
    def supported_model_types(self) -> List[str]:
        """Return list of model types this explainer supports."""
        pass


@dataclass
class ReasoningOutput:
    """
    Combined reasoning output integrating multiple explanation methods.
    
    Attributes:
        primary_explanation: Main explanation output
        supporting_explanations: Additional explanations from different methods
        reasoning_summary: Human-readable summary of the reasoning
        confidence_score: Overall confidence in the explanation (0-1)
    """
    primary_explanation: ExplanationOutput
    supporting_explanations: Optional[List[ExplanationOutput]] = None
    reasoning_summary: Optional[str] = None
    confidence_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate reasoning output after initialization."""
        if not isinstance(self.primary_explanation, ExplanationOutput):
            raise TypeError("primary_explanation must be an ExplanationOutput")
        
        if self.confidence_score is not None:
            if not 0 <= self.confidence_score <= 1:
                raise ValueError("confidence_score must be between 0 and 1")


__all__ = [
    'ExplainerInterface',
    'ExplanationOutput',
    'ReasoningOutput',
]
