# Predictive Claims Risk System with Explainability for Small Insurers

## Technical Description  
This project implements a machine learning system to predict the likelihood of insurance customers filing claims in the next year, combining advanced predictive modeling with interpretable explanations of risk factors. The system uses historical customer data (demographic, behavioral, and vehicular) to:  

1. **Temporal Risk Prediction:**  
   - Creates a target variable (`next_claim`) through temporal shifting, ensuring the model learns sequential patterns  
   - Maintains temporal integrity in train/test splits to prevent data leakage  

2. **Processing Pipeline:**  
   - Standardization of numerical features (age, credit score, claims history)  
   - Encoding of categorical variables (gender)  
   - Modeling with XGBoost optimized for imbalanced data via `scale_pos_weight`  

3. **SHAP Explainability:**  
   - Generates individual explanations for each prediction  
   - Identifies the top 3 risk factors per customer  
   - Quantifies the relative contribution of each factor as a percentage  

4. **Operational Output:**  
   - Risk classification into 4 tiers (Extreme/High/Moderate/Low)  
   - Top 20 high-risk clients ranking  
   - Technical report with validation metrics and model limitations  

## Key Differentiators  
- Combines quantitative prediction with qualitative explanation  
- Adapted for sequential temporal customer data  
- Self-explanatory system for regulatory decision-making  
- Proactive detection of extreme-risk clients (>95% probability)  

## Practical Application  
Enables insurers to prioritize preventive actions, customize insurance premiums, and comply with regulatory transparency requirements for risk models.  
