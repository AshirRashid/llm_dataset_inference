import numpy as np

def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two independent samples.
    
    Parameters:
    group1 (array-like): The first sample (e.g., heldout_train predictions/losses).
    group2 (array-like): The second sample (e.g., heldout_val predictions/losses).
    
    Returns:
    float: The Cohen's d effect size. A negative value indicates mean(group1) < mean(group2).
    """
    n1, n2 = len(group1), len(group2)
    
    if n1 <= 1 or n2 <= 1:
        return float('nan') # Cannot compute pooled std dev with insufficient samples

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0 # Means are identical, or variances are identically 0
        
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    d = (mean1 - mean2) / pooled_std
    return float(d)
