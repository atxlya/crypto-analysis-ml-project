📘 CRYPTO ANALYSIS & ML PREDICION DASHBOARD

A complete end-to-end cryptocurrency analysis and machine-learning prediction project built using **Python** and **Streamlit**.  
The app loads **90-day market-realistic datasets**, performs statistical analysis, calculates risk–reward scores, predicts next-day prices using **Random Forest**, compares multiple coins, simulates investments, generates alerts, and visualizes trends through interactive charts.

---

🚀 FEATURES

🔹 1. Data Loading & Cleaning
- Upload one or multiple CSV files  
- Cleans and normalizes each dataset  
- Handles dates, missing values, and inconsistent formatting  

🔹 2. Statistical Summary
- Average close price  
- Min/Max price  
- Daily returns  
- Volatility  
- Average volume  

🔹 3. Risk–Reward Score
- Sharpe-like score using annualized return vs volatility  
- Categorizes coins as **Good**, **Moderate**, or **Risky**

🔹 4. ML-Based Price Prediction
- Random Forest Regressor  
- Uses last 7 days of pricing  
- Predicts next-day closing price  
- Shows delta and percentage change  

🔹 5. Coin Comparison Dashboard
- Compares all uploaded coins  
- Ranks by risk–reward score  
- Highlights **Best Overall Coin to Invest In**  

🔹 6. Portfolio Simulation
- Enter any investment amount  
- Simulates buy-and-hold performance  
- Shows final value, profit, and return percentage  

🔹 7. Alert System
Triggers alerts if:
- Price drops >5%  
- Price jumps >8%  
- Volume spikes >2× average  

🔹 8. Interactive Charts
- Close price trend  
- Volume movement  
- Daily returns  

---

📂 Project Structure
.
├── crypto_app.py
├── BTC_90.csv
├── ETH_90.csv
├── DOGE_90.csv
├── DIGITAL_GOLD_90.csv
├── SHIBA_INU_90.csv
└── README.md
