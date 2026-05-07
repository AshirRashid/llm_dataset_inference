import numpy as np
from scipy.stats import chi2

def cohens_d(group1, group2):
    """Computes Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1: return float('nan')
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0: return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def fishers_method(p_values):
    """Combines independent p-values using Fisher's method."""
    p_values = np.array(p_values)
    p_values = np.clip(p_values, 1e-300, 1.0)
    statistic = -2 * np.sum(np.log(p_values))
    return chi2.sf(statistic, 2 * len(p_values))
