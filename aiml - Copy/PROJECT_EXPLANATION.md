# 🧬 YOUR SCHIZOPHRENIA GENETICS PROJECT - EXPLAINED SIMPLY

**Name:** ATHARVA JAIN  
**Roll No:** 2420030092  
**Team:** Group 24  
**Project:** Genetic Risk Factor Analysis in Schizophrenia Using Machine Learning

---

## 🎯 WHAT IS YOUR PROJECT ABOUT?

### **The Big Question:**
Can we use someone's DNA + life factors to predict their risk of developing schizophrenia?

### **Your Corrected Understanding:**

❌ **WRONG:** "People get schizophrenia → their genes change"

✅ **CORRECT:** "People are born with genetic variants → these increase their RISK of schizophrenia later"

**Key Points:**
1. **Genetics ≠ Destiny**: Having risk genes doesn't guarantee schizophrenia
2. **Polygenic = Many genes**: Not 1 "schizophrenia gene" - it's hundreds of small variants
3. **Environment matters**: Cannabis use, trauma, family history all play a role

---

## 📊 YOUR DATASET (SCZ2022)

### **What it Contains:**
- **NOT** individual people's genomes
- **IS** summary statistics from 76,755 patients + 243,649 healthy people
- **Shows** which genetic positions (SNPs) are linked to schizophrenia

### **Data Columns:**
```
SNP        = Genetic variant ID (e.g., rs1234567)
CHR        = Chromosome (1-22, X)
BP         = Position on chromosome
A1         = Risk allele (the "risky" version)
A2         = Normal allele
OR         = Odds Ratio (how much risk increases)
BETA       = Effect size (log of OR)
P          = Statistical significance
```

### **Example Row:**
```
SNP: rs123456
CHR: 6
P-value: 3.2e-9 (HIGHLY significant)
OR: 1.15 (15% increased risk per copy)
→ This SNP matters for schizophrenia!
```

---

## 🔬 YOUR PROJECT - WHAT IT DOES (3 PHASES)

### **PHASE 1: Load Real GWAS Data** 📊

**What happens:**
1. Load 9.5 million SNPs from the SCZ2022 dataset
2. Find which ones are "genome-wide significant" (P < 5e-8)
3. Create Manhattan plot (shows significant variants across chromosomes)
4. Create Q-Q plot (checks data quality)

**Output:**
- Top 10 most important schizophrenia risk genes
- Visualization of genetic risk across the genome
- Focus on 22q11.2 deletion (major risk region)

**Files generated:**
- `manhattan_plot.png`
- `qq_plot.png`
- `top_risk_snps.csv`

---

### **PHASE 2: Calculate Polygenic Risk Scores (PRS)** 🧬

**What is PRS?**
Think of it like a **credit score**, but for disease risk!

**Formula:**
```
PRS = (SNP1 × weight1) + (SNP2 × weight2) + ... + (SNP100 × weight100)

Where:
- SNP value = 0, 1, or 2 (number of risk alleles)
- Weight = BETA from GWAS (how much this SNP matters)
```

**Example Patient:**
```
Patient_001:
- rs123456: 2 copies (homozygous risk)  → +0.15
- rs789012: 1 copy (heterozygous)       → +0.08
- rs345678: 0 copies (no risk)          → +0.00
...
Total PRS = 1.34 (normalized: +0.8 standard deviations)
```

**What your code does:**
1. Select top 100 significant SNPs
2. Create 5,000 **synthetic patients** with realistic genotypes
3. Calculate PRS for each patient
4. Add environmental factors:
   - **Family history** (1x, 3x, 10x, 50x risk multipliers)
   - Cannabis use (+0.5 risk)
   - Urban birth (+0.3 risk)
   - Prenatal infection (+0.4 risk)
   - Childhood trauma (+0.4 risk)
   - Paternal age (older = higher risk)
5. Assign risk categories: Low, Moderate, High, Very High
6. Assign case/control status (1.2% cases in your data)

**Files generated:**
- `synthetic_patients_with_prs.csv` (5,000 patients)
- `prs_snp_set.csv` (100 SNPs used)
- `phase2_prs_analysis.png` (6 visualizations)

---

### **PHASE 3: Train XGBoost Machine Learning Model** 🤖

**What is XGBoost?**
A powerful AI algorithm that learns patterns to make predictions.

**What it learns:**
Given these 8 features:
1. PRS_normalized (genetic risk score)
2. Cannabis_Use (0 or 1)
3. Urban_Birth (0 or 1)
4. Prenatal_Infection (0 or 1)
5. Childhood_Trauma (0 or 1)
6. Paternal_Age (years)
7. Environmental_Risk (combined environmental score)
8. FH_RiskMultiplier (family history: 1, 3, 10, or 50)

**Predict:** Is this person a case (schizophrenia) or control (healthy)?

**Your Results:**
```
Training:   3,500 patients (44 cases)
Validation:   500 patients (6 cases)
Test:       1,000 patients (12 cases)

Performance:
✓ Accuracy:  98.2%
✓ ROC-AUC:   87.3% (excellent!)
✓ Precision: 20.0% (conservative predictions)
✓ Recall:    16.7% (finds some cases)
```

**Feature Importance (What matters most?):**
```
1. Family History         53.5%  ⭐⭐⭐⭐⭐ (HUGE!)
2. Urban Birth            12.1%  ⭐⭐
3. PRS (Genetics)          9.4%  ⭐⭐
4. Cannabis Use            6.0%  ⭐
5. Environmental Risk      5.7%  ⭐
6. Paternal Age            5.0%  ⭐
7. Prenatal Infection      4.2%  
8. Childhood Trauma        4.0%
```

**Key Finding:** **Family history is 5x more important than genetics alone!**

**Files generated:**
- `xgboost_scz_risk_model.json` (trained model)
- `feature_importance.csv`
- `phase3_model_evaluation.png` (8 plots showing model performance)

---

## 🎓 HOW TO EXPLAIN THIS TO YOUR PROFESSOR

### **One-Sentence Summary:**
"We use real genetic data from 76,000 schizophrenia patients to build a machine learning model that predicts disease risk by combining DNA variants (polygenic risk scores) with family history and environmental factors."

### **Key Points to Mention:**

1. **Real-World Dataset:**
   - PGC SCZ2022: 76,755 cases, 243,649 controls
   - Published in Nature Genetics (2022)
   - 9.5 million genetic variants analyzed

2. **Polygenic Risk Score:**
   - Schizophrenia isn't caused by ONE gene
   - We combine 100 significant variants into a single score
   - Like summing up many small risks into a total risk

3. **Gene-Environment Interaction:**
   - Genetics alone isn't enough
   - Family history is the #1 predictor (53.5%)
   - Environmental factors (cannabis, trauma, urban birth) add additional risk

4. **Machine Learning (XGBoost):**
   - Trained on 5,000 synthetic patients
   - Achieves 87% ROC-AUC (excellent discrimination)
   - Can identify high-risk individuals for early intervention

5. **Ethical Considerations:**
   - This is EDUCATIONAL - not for clinical use
   - Genetic risk ≠ diagnosis
   - Requires professional genetic counseling in real scenarios

---

## 🚀 GOOGLE COLAB INSTRUCTIONS

### **Step 1: Download the Data**
1. Go to: https://figshare.com/articles/dataset/scz2022/19426775
2. Click "Download" on `daner_PGC_SCZ_w3_90_0418b_ukbbdedupe.gz`
3. Wait (it's 1.2 GB)

### **Step 2: Upload to Colab**
1. Open Google Colab
2. Click folder icon (left sidebar)
3. Click upload button
4. Select the `.gz` file
5. Wait 5-10 minutes for upload

### **Step 3: Run the Code**
1. Copy `colab_starter.py` code into Colab
2. Install packages:
   ```python
   !pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
   ```
3. Run all cells
4. Wait 2-3 minutes for data loading

### **What You'll Get:**
- Manhattan plot showing significant variants
- Q-Q plot checking data quality
- Top 10 schizophrenia risk SNPs
- 22q11.2 deletion region analysis

---

## 📚 TERMS YOU SHOULD KNOW

| Term | Meaning | Example |
|------|---------|---------|
| **SNP** | Single Nucleotide Polymorphism - a single letter difference in DNA | rs123456: A→G |
| **GWAS** | Genome-Wide Association Study - finds genetic variants linked to diseases | PGC SCZ2022 |
| **P-value** | Statistical significance (lower = more confident) | P < 5e-8 = genome-wide significant |
| **Odds Ratio (OR)** | How much a variant increases disease risk | OR = 1.15 means 15% higher risk |
| **BETA** | Log of odds ratio (effect size) | BETA = log(1.15) = 0.14 |
| **PRS** | Polygenic Risk Score - combined genetic risk from many variants | Score: +1.2 SD above average |
| **Allele** | One version of a gene | A1 = risk allele, A2 = normal |
| **22q11.2** | Major schizophrenia risk region on chromosome 22 | 30x increased risk if deleted |
| **ROC-AUC** | Model performance metric (0.5 = random, 1.0 = perfect) | 0.87 = excellent |
| **XGBoost** | Gradient Boosting ML algorithm | Popular for tabular data |

---

## ✅ CHECKLIST FOR SUBMISSION

- [ ] Downloaded SCZ2022 dataset (1.2 GB)
- [ ] Ran code in Google Colab successfully
- [ ] Generated all 3 phases of visualizations
- [ ] Understand PRS calculation
- [ ] Can explain XGBoost model results
- [ ] Know top risk genes (from Manhattan plot)
- [ ] Understand family history is #1 predictor
- [ ] Prepared to discuss ethical considerations
- [ ] Ready to cite Trubetskoy et al., Nature Genetics (2022)

---

## 🎯 FINAL SUMMARY

**What you built:**
A complete bioinformatics + machine learning pipeline that:
1. Loads real genetic data (9.5M SNPs)
2. Calculates polygenic risk scores
3. Simulates patients with realistic genetics
4. Trains an AI model to predict schizophrenia risk
5. Identifies that family history + genetics + environment all matter

**Why it matters:**
- **Precision medicine:** Personalized risk assessment
- **Early intervention:** Identify high-risk individuals
- **Research:** Understand schizophrenia genetics
- **Education:** Learn computational psychiatry

**Your contribution:**
Demonstrating how modern genomics + AI can advance mental health care (responsibly and ethically).

---

**Good luck with your project! 🚀**

**Questions?** Review this document or check your notebook visualizations!
