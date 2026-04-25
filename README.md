# unbiased-ai-detection
# Unbiased AI Detection System

A Streamlit-based tool to **detect bias and proxy features in datasets** before training machine learning models.  
This project helps ensure **fair, transparent, and responsible AI systems**.

## 🚀 Features
### 📊 Bias Detection
- Computes key fairness metrics:
  - **Demographic Parity Difference**
  - **Disparate Impact (80% rule)**
- Automatically detects:
  - Biased groups
  - Imbalanced outcomes across sensitive attributes
---
### 🔗 Proxy Feature Detection
- Identifies features indirectly correlated with sensitive attributes
- Uses:
  - Mutual Information
  - Correlation (for numeric features)
  - Cramér’s V (for categorical features)
---
### 📈 Visualizations
- Group-wise positive rate comparison
- Fairness summary charts
- Proxy feature heatmaps
---
### 🛠️ Actionable Recommendations
Provides mitigation strategies such as:
- Reweighing
- Disparate Impact Remover
- Proxy feature removal
- Threshold optimization
---
### 🤖 AI Explanation (Optional)
- Integrates with **Google Gemini API**
- Generates human-friendly explanations of:
  - Detected bias
  - Root causes
  - Fix strategies
---
## 🧠 How It Works
1. Upload dataset (CSV)
2. Select:
   - Target column
   - Sensitive features (e.g., gender, caste, income group)
3. System:
   - Preprocesses data
   - Computes fairness metrics
   - Detects proxy features
4. Displays:
   - Bias indicators
   - Visual insights
   - Mitigation suggestions
---
## 📦 Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/unbiased-ai-detection.git
cd unbiased-ai-detection 
```
### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
▶️ Run the App
```bash
streamlit run unbiased_ai_detector.py
```
### 🔐 Gemini API (Optional)
To enable AI explanations:
1. Use Streamlit secret
   - Create .streamlit/secrets.toml    
   - GEMINI_API_KEY = "your_api_key"
2. Enter manually in UI
