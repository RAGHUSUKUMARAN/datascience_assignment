import numpy as np
from scipy.stats import chi2_contingency, chi2

# Observed contingency table (rows = satisfaction levels, columns = device types)
obs = np.array([
    [50, 70],   # Very Satisfied
    [80,100],   # Satisfied
    [60,90],    # Neutral
    [30,50],    # Unsatisfied
    [20,50]     # Very Unsatisfied
])

chi2_stat, p_value, dof, expected = chi2_contingency(obs, correction=False)
crit_value = chi2.ppf(0.95, df=dof)

print("Chi-square statistic:", round(chi2_stat, 3))
print("Degrees of freedom:", dof)
print("p-value:", round(p_value, 4))
print("Critical value (alpha=0.05):", round(crit_value, 3))
print("\nExpected counts:\n", expected)
