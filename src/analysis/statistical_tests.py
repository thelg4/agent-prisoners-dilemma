from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass
import logging

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    significant: bool
    interpretation: str

class StatisticalTests:
    """Statistical analysis for tournament results"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
    
    def compare_cooperation_rates(
        self, 
        group1_rates: List[float], 
        group2_rates: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> StatisticalResult:
        """Compare cooperation rates between two groups"""
        
        # Convert to numpy arrays
        rates1 = np.array(group1_rates)
        rates2 = np.array(group2_rates)
        
        # Perform two-sample t-test
        statistic, p_value = stats.ttest_ind(rates1, rates2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(rates1) - 1) * np.var(rates1, ddof=1) + 
                             (len(rates2) - 1) * np.var(rates2, ddof=1)) / 
                            (len(rates1) + len(rates2) - 2))
        
        cohens_d = (np.mean(rates1) - np.mean(rates2)) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for difference in means
        se_diff = pooled_std * np.sqrt(1/len(rates1) + 1/len(rates2))
        df = len(rates1) + len(rates2) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        mean_diff = np.mean(rates1) - np.mean(rates2)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Interpretation
        significant = p_value < self.alpha
        
        if significant:
            direction = "higher" if mean_diff > 0 else "lower"
            interpretation = (f"{group1_name} has significantly {direction} cooperation rates "
                            f"than {group2_name} (p={p_value:.4f}, d={cohens_d:.3f})")
        else:
            interpretation = (f"No significant difference in cooperation rates between "
                            f"{group1_name} and {group2_name} (p={p_value:.4f})")
        
        return StatisticalResult(
            test_name="Two-sample t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            significant=significant,
            interpretation=interpretation
        )
    
    def compare_score_distributions(
        self,
        group1_scores: List[float],
        group2_scores: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2"
    ) -> StatisticalResult:
        """Compare score distributions using Mann-Whitney U test"""
        
        scores1 = np.array(group1_scores)
        scores2 = np.array(group2_scores)
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(scores1), len(scores2)
        effect_size = (2 * statistic) / (n1 * n2) - 1
        
        significant = p_value < self.alpha
        
        if significant:
            median1, median2 = np.median(scores1), np.median(scores2)
            direction = "higher" if median1 > median2 else "lower"
            interpretation = (f"{group1_name} has significantly {direction} score distribution "
                            f"than {group2_name} (p={p_value:.4f}, r={effect_size:.3f})")
        else:
            interpretation = (f"No significant difference in score distributions between "
                            f"{group1_name} and {group2_name} (p={p_value:.4f})")
        
        return StatisticalResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=None,
            significant=significant,
            interpretation=interpretation
        )
    
    def test_cooperation_trend(self, cooperation_rates: List[float]) -> StatisticalResult:
        """Test for trend in cooperation rates over time"""
        
        x = np.arange(len(cooperation_rates))
        y = np.array(cooperation_rates)
        
        # Spearman correlation for monotonic trend
        correlation, p_value = stats.spearmanr(x, y)
        
        significant = p_value < self.alpha
        
        if significant:
            trend_direction = "increasing" if correlation > 0 else "decreasing"
            interpretation = (f"Significant {trend_direction} trend in cooperation rates "
                            f"(ρ={correlation:.3f}, p={p_value:.4f})")
        else:
            interpretation = f"No significant trend in cooperation rates (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name="Spearman correlation",
            statistic=correlation,
            p_value=p_value,
            effect_size=correlation,
            confidence_interval=None,
            significant=significant,
            interpretation=interpretation
        )
    
    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = None
    ) -> Dict[str, float]:
        """Calculate statistical power for given parameters"""
        
        if alpha is None:
            alpha = self.alpha
        
        # For two-sample t-test
        from statsmodels.stats.power import ttest_power
        
        power = ttest_power(effect_size, sample_size/2, alpha, alternative='two-sided')
        
        return {
            "effect_size": effect_size,
            "sample_size": sample_size,
            "alpha": alpha,
            "power": power,
            "adequate_power": power >= 0.8
        }
    
    def multiple_comparisons_correction(
        self, 
        p_values: List[float], 
        method: str = "bonferroni"
    ) -> Tuple[List[bool], List[float]]:
        """Apply multiple comparisons correction"""
        
        from statsmodels.stats.multitest import multipletests
        
        rejected, p_corrected, _, _ = multipletests(
            p_values, 
            alpha=self.alpha, 
            method=method
        )
        
        return rejected.tolist(), p_corrected.tolist()