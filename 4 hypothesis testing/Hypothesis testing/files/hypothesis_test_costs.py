# hypothesis_test_costs.py
import math

# Given data
x_bar = 3050.0       # sample mean weekly cost (Rs)
n = 25               # sample size
mu_model = 1000 + 5 * 600   # theoretical mean from model: 1000 + 5*600
sigma_X = 25.0       # sd of X (units)
sigma_W = 5.0 * sigma_X     # sd of W (cost), since W = 1000 + 5X

# Calculations
se = sigma_W / math.sqrt(n)
z_stat = (x_bar - mu_model) / se

# Critical value for right-tailed test at alpha=0.05
alpha = 0.05
try:
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha)
except Exception:
    # fallback: standard critical value for 0.05 right-tailed
    z_crit = 1.645

# Decision
reject = z_stat > z_crit

# Output
print("Given values:")
print(f" sample mean (x̄) = {x_bar}")
print(f" model mean (μ)   = {mu_model}")
print(f" sigma_W (σ)      = {sigma_W}")
print(f" sample size (n)  = {n}")
print()
print("Calculations:")
print(f" standard error (σ/√n) = {se:.4f}")
print(f" test statistic Z       = {z_stat:.4f}")
print(f" critical value (right-tailed, α={alpha}) z_crit = {z_crit:.4f}")
print()
if reject:
    print("Decision: Reject H0 in favor of H1 (evidence that mean cost is higher).")
else:
    print("Decision: Fail to reject H0 (no evidence that mean cost is higher).")
    # Extra helpful note:
    print("Note: The sample mean is lower than the theoretical mean; the test statistic is negative,")
    print("so the data provide no support for the claim that costs are higher (they appear lower).")
