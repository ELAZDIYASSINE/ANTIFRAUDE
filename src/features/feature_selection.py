#!/usr/bin/env python3
"""
Automated Feature Selection for Anti-Fraud Detection
Features: Variance threshold, correlation analysis, feature importance
"""

import os
import sys
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, when, lit
from pyspark.sql.stat import Correlation
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection"""
    selected_features: List[str]
    removed_features: List[str]
    variance_threshold: float
    correlation_threshold: float
    selection_reason: str


class FeatureSelector:
    """Automated Feature Selection"""
    
    def __init__(self, spark: SparkSession):
        """Initialize feature selector"""
        self.spark = spark
        self.variance_threshold = 0.01
        self.correlation_threshold = 0.95
        
    def calculate_variance(self, df: DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """Calculate variance for each feature"""
        logger.info("Calculating feature variances...")
        
        variances = {}
        for feature in feature_cols:
            try:
                # Calculate variance
                variance = df.select(F.variance(col(feature))).collect()[0][0]
                if variance is None:
                    variance = 0.0
                variances[feature] = variance
                logger.debug(f"{feature}: variance = {variance:.6f}")
            except Exception as e:
                logger.warning(f"Could not calculate variance for {feature}: {e}")
                variances[feature] = 0.0
        
        return variances
    
    def remove_low_variance_features(self, df: DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
        """Remove features with variance below threshold"""
        logger.info(f"Removing low variance features (threshold: {self.variance_threshold})")
        
        variances = self.calculate_variance(df, feature_cols)
        
        selected = []
        removed = []
        
        for feature, variance in variances.items():
            if variance >= self.variance_threshold:
                selected.append(feature)
            else:
                removed.append(feature)
                logger.info(f"  Removed {feature}: variance = {variance:.6f}")
        
        logger.info(f"Selected: {len(selected)}, Removed: {len(removed)}")
        return selected, removed
    
    def calculate_correlation_matrix(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """Calculate correlation matrix for features"""
        logger.info("Calculating correlation matrix...")
        
        # Convert to pandas for correlation calculation (more efficient)
        try:
            import pandas as pd
            pdf = df.select(feature_cols).toPandas()
            correlation_matrix = pdf.corr()
            
            logger.info(f"Correlation matrix calculated: {len(feature_cols)}x{len(feature_cols)}")
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def find_highly_correlated_features(self, correlation_matrix, threshold: float) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs"""
        logger.info(f"Finding highly correlated features (threshold: {threshold})")
        
        if correlation_matrix is None:
            return []
        
        correlated_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                correlation = abs(correlation_matrix.iloc[i, j])
                
                if correlation >= threshold:
                    correlated_pairs.append((feature1, feature2, correlation))
                    logger.info(f"  {feature1} <-> {feature2}: {correlation:.4f}")
        
        return correlated_pairs
    
    def remove_correlated_features(self, df: DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
        """Remove highly correlated features"""
        logger.info("Removing highly correlated features...")
        
        correlation_matrix = self.calculate_correlation_matrix(df, feature_cols)
        correlated_pairs = self.find_highly_correlated_features(correlation_matrix, self.correlation_threshold)
        
        # Remove one feature from each correlated pair
        to_remove = set()
        for feature1, feature2, correlation in correlated_pairs:
            # Remove the second feature (arbitrary choice, could be improved)
            if feature2 not in to_remove:
                to_remove.add(feature2)
                logger.info(f"  Removing {feature2} (correlated with {feature1}: {correlation:.4f})")
        
        selected = [f for f in feature_cols if f not in to_remove]
        removed = list(to_remove)
        
        logger.info(f"Selected: {len(selected)}, Removed: {len(removed)}")
        return selected, removed
    
    def calculate_feature_importance(self, df: DataFrame, target_col: str, feature_cols: List[str]) -> Dict[str, float]:
        """Calculate feature importance using correlation with target"""
        logger.info("Calculating feature importance (correlation with target)...")
        
        importances = {}
        
        for feature in feature_cols:
            try:
                correlation = df.stat.corr(feature, target_col)
                if correlation is None:
                    correlation = 0.0
                importances[feature] = abs(correlation)
            except Exception as e:
                logger.warning(f"Could not calculate importance for {feature}: {e}")
                importances[feature] = 0.0
        
        # Sort by importance
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Top 10 features by importance:")
        for i, (feature, importance) in enumerate(list(sorted_importances.items())[:10]):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        return sorted_importances
    
    def select_top_features(self, df: DataFrame, target_col: str, feature_cols: List[str], 
                          top_n: int = 50) -> FeatureSelectionResult:
        """Select top N features based on importance"""
        logger.info(f"Selecting top {top_n} features...")
        
        # Step 1: Remove low variance features
        variance_selected, variance_removed = self.remove_low_variance_features(df, feature_cols)
        
        # Step 2: Remove correlated features
        correlation_selected, correlation_removed = self.remove_correlated_features(df, variance_selected)
        
        # Step 3: Calculate feature importance
        importances = self.calculate_feature_importance(df, target_col, correlation_selected)
        
        # Step 4: Select top N features
        top_features = list(importances.keys())[:top_n]
        removed_features = variance_removed + correlation_removed
        
        result = FeatureSelectionResult(
            selected_features=top_features,
            removed_features=removed_features,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            selection_reason=f"Top {top_n} features after variance and correlation filtering"
        )
        
        logger.info(f"✅ Feature selection complete: {len(top_features)} selected, {len(removed_features)} removed")
        
        return result
    
    def apply_feature_selection(self, df: DataFrame, result: FeatureSelectionResult) -> DataFrame:
        """Apply feature selection to dataframe"""
        logger.info("Applying feature selection...")
        
        selected_cols = result.selected_features + ['isFraud']  # Keep target column
        df_selected = df.select(selected_cols)
        
        logger.info(f"✅ Feature selection applied: {len(df_selected.columns)} columns")
        
        return df_selected
    
    def generate_feature_selection_report(self, result: FeatureSelectionResult) -> str:
        """Generate feature selection report"""
        report = "\n" + "=" * 60
        report += "\n📊 FEATURE SELECTION REPORT"
        report += "\n" + "=" * 60
        report += f"\nVariance Threshold: {result.variance_threshold}"
        report += f"\nCorrelation Threshold: {result.correlation_threshold}"
        report += f"\nSelection Reason: {result.selection_reason}"
        report += f"\n\nSelected Features: {len(result.selected_features)}"
        report += f"\nRemoved Features: {len(result.removed_features)}"
        report += "\n\nTop 20 Selected Features:"
        
        for i, feature in enumerate(result.selected_features[:20]):
            report += f"\n  {i+1}. {feature}"
        
        report += "\n\nRemoved Features:"
        for feature in result.removed_features[:10]:
            report += f"\n  - {feature}"
        if len(result.removed_features) > 10:
            report += f"\n  ... and {len(result.removed_features) - 10} more"
        
        report += "\n" + "=" * 60
        
        return report


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName("FeatureSelection").getOrCreate()
        
        # Load feature table
        df = spark.read.format("delta").load("data/feature_store/transaction_features")
        
        # Get feature columns (exclude non-numeric columns)
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.typeName() in ['integer', 'double', 'float', 'long']]
        non_numeric_cols = [f.name for f in df.schema.fields if f.dataType.typeName() not in ['integer', 'double', 'float', 'long']]
        
        # Remove target from features
        feature_cols = [c for c in numeric_cols if c != 'isFraud']
        
        logger.info(f"Total numeric features: {len(feature_cols)}")
        logger.info(f"Non-numeric columns: {non_numeric_cols}")
        
        # Run feature selection
        selector = FeatureSelector(spark)
        result = selector.select_top_features(df, 'isFraud', feature_cols, top_n=50)
        
        # Apply selection
        df_selected = selector.apply_feature_selection(df, result)
        
        # Save selected features
        df_selected.write.format("delta").mode("overwrite").save("data/feature_store/selected_features")
        
        # Generate report
        report = selector.generate_feature_selection_report(result)
        print(report)
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Feature selection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
