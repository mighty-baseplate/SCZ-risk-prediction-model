# ===============================================================================
# SCHIZOPHRENIA GWAS ANALYSIS - GOOGLE COLAB VERSION
# For use with real PGC SCZ2022 dataset
# ===============================================================================

"""
BEFORE RUNNING IN COLAB:
1. Upload daner_PGC_SCZ_w3_90_0418b_ukbbdedupe.gz to Colab
2. Install packages: !pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
3. Run this script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD REAL GWAS DATA
# =============================================================================

print("="*80)
print("LOADING REAL PGC SCHIZOPHRENIA GWAS DATA (2022)")
print("="*80)

# Path to your uploaded file in Colab
GWAS_FILE = "daner_PGC_SCZ_w3_90_0418b_ukbbdedupe.gz"

# Column names for PGC data
column_names = ['CHR', 'SNP', 'BP', 'A1', 'A2', 'FRQ_A', 'FRQ_U', 'INFO', 
                'OR', 'SE', 'P', 'ngt', 'Direction', 'HetISqt', 'HetChiSq', 'HetDf', 'HetPVa']

print("\n📊 Loading GWAS data (this takes 2-3 minutes for 9.5M SNPs)...")

try:
    # Load the data
    # For initial testing, use nrows=100000 to load faster
    # Remove nrows parameter to load full dataset
    df = pd.read_csv(
        GWAS_FILE, 
        sep='\t',  # tab-delimited
        compression='gzip',
        comment='#',  # skip comment lines
        nrows=100000  # Remove this line to load ALL 9.5M SNPs
    )
    
    print(f"✅ Loaded {len(df):,} SNPs")
    
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {GWAS_FILE}")
    print("   Please upload the file to Google Colab first!")
    print("   Download from: https://figshare.com/articles/dataset/scz2022/19426775")
    exit()

# =============================================================================
# STEP 2: UNDERSTAND THE DATA
# =============================================================================

print(f"\n📋 Data Overview:")
print(f"   Total SNPs: {len(df):,}")
print(f"   Columns: {list(df.columns)}")
print(f"\n🔬 Column Meanings:")
print("   CHR: Chromosome (1-22, X)")
print("   SNP: SNP identifier (e.g., rs123456)")
print("   BP: Base pair position on chromosome")
print("   A1: Effect allele (risk-increasing version)")
print("   A2: Reference allele")
print("   OR: Odds Ratio (how much A1 increases risk)")
print("   P: P-value (statistical significance)")
print("   SE: Standard Error")

print(f"\n📊 First few rows:")
print(df.head())

# =============================================================================
# STEP 3: CALCULATE BETA FROM ODDS RATIO
# =============================================================================

print(f"\n🧮 Converting Odds Ratios to BETA values...")

# BETA = log(OR)
df['BETA'] = np.log(df['OR'])

# Remove missing/invalid values
df_clean = df.dropna(subset=['P', 'BETA', 'SE']).copy()
df_clean = df_clean[(df_clean['P'] > 0) & (df_clean['P'] <= 1)]

print(f"   After quality control: {len(df_clean):,} SNPs")

# =============================================================================
# STEP 4: FIND SIGNIFICANT GENETIC VARIANTS
# =============================================================================

print(f"\n🔍 Finding Genome-Wide Significant SNPs...")

# Genome-wide significance threshold
GWAS_THRESHOLD = 5e-8

sig_snps = df_clean[df_clean['P'] < GWAS_THRESHOLD]

print(f"   Genome-wide significant (P < 5e-8): {len(sig_snps):,} SNPs")
print(f"   Suggestive (P < 1e-5): {(df_clean['P'] < 1e-5).sum():,} SNPs")

# =============================================================================
# STEP 5: TOP RISK SNPS
# =============================================================================

print(f"\n🔝 TOP 10 SCHIZOPHRENIA RISK VARIANTS:")
print("="*80)

top_snps = df_clean.nsmallest(10, 'P')

for idx, (_, row) in enumerate(top_snps.iterrows(), 1):
    direction = "↑ INCREASED RISK" if row['BETA'] > 0 else "↓ PROTECTIVE"
    chr_pos = f"Chr{row['CHR']}:{row['BP']}"
    
    print(f"{idx:2d}. {row['SNP']:<15} {chr_pos:<20} P={row['P']:.2e}  OR={row['OR']:.4f}  {direction}")

# Save results
top_snps.to_csv('top_schizophrenia_risk_snps.csv', index=False)
print(f"\n💾 Saved: top_schizophrenia_risk_snps.csv")

# =============================================================================
# STEP 6: CHROMOSOME 22q11.2 DELETION REGION
# =============================================================================

print(f"\n🧬 Checking 22q11.2 Deletion Region (Major SCZ Risk Locus)...")

# 22q11.2 is roughly 18-21 Mb on chromosome 22
chr22_region = df_clean[(df_clean['CHR'] == 22) & 
                         (df_clean['BP'] >= 18000000) & 
                         (df_clean['BP'] <= 21000000)]

if len(chr22_region) > 0:
    sig_in_region = chr22_region[chr22_region['P'] < 5e-8]
    print(f"   SNPs in 22q11.2 region: {len(chr22_region):,}")
    print(f"   Significant SNPs: {len(sig_in_region):,}")
    
    if len(sig_in_region) > 0:
        print(f"\n   Top variant in 22q11.2:")
        top_22q = sig_in_region.nsmallest(1, 'P').iloc[0]
        print(f"   {top_22q['SNP']} - P={top_22q['P']:.2e}")

# =============================================================================
# STEP 7: MANHATTAN PLOT
# =============================================================================

print(f"\n📊 Creating Manhattan Plot...")

# Sample if too large
if len(df_clean) > 500000:
    df_plot = df_clean.sample(n=500000, random_state=42)
else:
    df_plot = df_clean.copy()

df_plot['-log10P'] = -np.log10(df_plot['P'])
df_plot = df_plot[np.isfinite(df_plot['-log10P'])]

fig, ax = plt.subplots(figsize=(16, 6))

# Plot each chromosome in different color
colors = plt.cm.tab20(np.linspace(0, 1, 22))
for chrom in range(1, 23):
    chr_data = df_plot[df_plot['CHR'] == chrom]
    if len(chr_data) > 0:
        ax.scatter(chr_data.index, chr_data['-log10P'], 
                  c=[colors[chrom-1]], alpha=0.6, s=2, label=f'Chr{chrom}')

# Significance lines
ax.axhline(-np.log10(5e-8), color='red', linestyle='--', linewidth=2, 
           label='Genome-wide significance (5e-8)')
ax.axhline(-np.log10(1e-5), color='orange', linestyle='--', linewidth=2, 
           label='Suggestive (1e-5)')

ax.set_xlabel('SNP Index', fontsize=12, fontweight='bold')
ax.set_ylabel('-log₁₀(P-value)', fontsize=12, fontweight='bold')
ax.set_title('Manhattan Plot: PGC Schizophrenia GWAS 2022\n76,755 Cases vs 243,649 Controls', 
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('manhattan_plot_real_data.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: manhattan_plot_real_data.png")
plt.show()

# =============================================================================
# STEP 8: Q-Q PLOT
# =============================================================================

print(f"\n📊 Creating Q-Q Plot...")

# Sample for Q-Q plot
if len(df_clean) > 500000:
    sample_df = df_clean.sample(n=500000, random_state=42)
else:
    sample_df = df_clean

observed_p = np.sort(sample_df['P'].values)
n = len(observed_p)
expected_p = np.arange(1, n+1) / (n + 1)

obs_log = -np.log10(observed_p)
exp_log = -np.log10(expected_p)

# Calculate genomic inflation factor (lambda)
from scipy.stats import chi2
chisq = chi2.ppf(1 - sample_df['P'], df=1)
lambda_gc = np.median(chisq) / chi2.ppf(0.5, df=1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(exp_log, obs_log, alpha=0.6, s=3, c='steelblue')

max_val = max(exp_log.max(), obs_log.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Expected (null)')

ax.set_xlabel('Expected -log₁₀(P)', fontsize=12, fontweight='bold')
ax.set_ylabel('Observed -log₁₀(P)', fontsize=12, fontweight='bold')
ax.set_title(f'Q-Q Plot: P-value Inflation Test\nλ_GC = {lambda_gc:.3f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('qq_plot_real_data.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: qq_plot_real_data.png")
print(f"   Genomic inflation factor (λ): {lambda_gc:.3f}")
print(f"   (λ close to 1.0 = good quality data)")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("✅ PHASE 1 COMPLETE - REAL GWAS DATA LOADED!")
print("="*80)

print(f"""
📊 SUMMARY:
   ✓ Analyzed {len(df_clean):,} high-quality SNPs
   ✓ Found {len(sig_snps):,} genome-wide significant variants
   ✓ Identified top risk variants on Chr 1-22
   ✓ Checked 22q11.2 deletion region
   ✓ Created Manhattan plot
   ✓ Created Q-Q plot (λ = {lambda_gc:.3f})

📁 FILES GENERATED:
   ✓ top_schizophrenia_risk_snps.csv
   ✓ manhattan_plot_real_data.png
   ✓ qq_plot_real_data.png

🎯 NEXT STEPS:
   1. Use these significant SNPs to calculate Polygenic Risk Scores
   2. Generate synthetic patient genotypes
   3. Train XGBoost model to predict schizophrenia risk
   
   Your GWAS data is ready for Phase 2!
""")

print("="*80)
