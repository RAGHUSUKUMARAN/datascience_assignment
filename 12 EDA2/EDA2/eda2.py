import pandas as pd

# Load dataset
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\adult_with_headers.csv"
adult_df = pd.read_csv(file_path)

# Collect exploration results
output_lines = []
output_lines.append(f"Dataset shape: {adult_df.shape}\n")
output_lines.append("\nData types:\n")
output_lines.append(str(adult_df.dtypes))
output_lines.append("\n\nFirst 5 rows:\n")
output_lines.append(str(adult_df.head()))
output_lines.append("\n\nMissing values:\n")
output_lines.append(str(adult_df.isnull().sum()))

# Check '?' as missing placeholders
output_lines.append("\n\nUnique values in workclass:\n")
output_lines.append(str(adult_df['workclass'].unique()))
output_lines.append("\nUnique values in occupation:\n")
output_lines.append(str(adult_df['occupation'].unique()))
output_lines.append("\nUnique values in native_country:\n")
output_lines.append(str(adult_df['native_country'].unique()))

# Save results to text file in same folder
out_path = r"D:\DATA SCIENCE\ASSIGNMENTS\12 EDA2\EDA2\eda_summary.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print("✅ Exploration summary saved to:", out_path)
