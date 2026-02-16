# 🧹 1. Data Preprocessing & EDA (Notebook 2)

This notebook performs:

  * Data cleaning (handling missing values, duplicates)
  * Datetime conversion for time-series
  * Aggregation of sales by date
  * Trend visualization
  * Seasonality exploration
  * Outlier detection
  * Feature extraction (day, month, year, week)

       ### Example Code Snippets
        import pandas as pd
        df = pd.read_csv('data/raw_data.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df.head()
    
      ### Aggregate daily data 
        daily_sales = df.groupby('Order Date').agg({
        'Sales': 'sum',
        'Quantity': 'sum',
        'Profit': 'sum'}).reset_index()
    
      ### Explore data visually
        # Histograms for numerical features
        df.hist(bins=50, figsize=(20, 15))
        plt.show()

        # Box plots for numerical features
        for col in df.select_dtypes(include=np.number).columns:
        sns.boxplot(x=df[col])
        plt.show()

        # Correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[['Profit','Discount','Quantity','Sales']].corr(), annot=True, cmap='coolwarm')
        plt.show()

<img width="1384" height="558" alt="image" src="https://github.com/user-attachments/assets/cb0f11ed-a080-4199-8e4f-cda52e47e3b9" />

<img width="486" height="582" alt="image" src="https://github.com/user-attachments/assets/3a4c058a-863e-485a-8c8f-90b9f13c4257" />

<img width="472" height="415" alt="Screenshot 2026-02-16 at 20 41 57" src="https://github.com/user-attachments/assets/e6cf35fb-988e-495c-9012-71689d528b8c" />

### Check Autocorrelation for Sales and Proft  
   
<img width="472" height="427" alt="Screenshot 2026-02-16 at 20 48 07" src="https://github.com/user-attachments/assets/d8e80784-9241-4340-be02-59486a7c6308" />

<img width="472" height="337" alt="Screenshot 2026-02-16 at 20 48 36" src="https://github.com/user-attachments/assets/f15e0178-a896-4c0c-b310-8b5500a8769a" />

      Autocorrelation for both Sales and Profit is not significant.

    This implies:
  
    There is no meaningful relationship between values at different lags.

    The series does not exhibit strong seasonality or repeated patterns in the examined lags (up to 24 months).

### Check ADF Test (Augmented Dickey-Fuller)

**Purpose:** Checking wheter the data is stationer or not. This also usefull to know before we create the forecasting model **.

- If **p-value < 0.05** → **data is stationer**
- If **p-value > 0.05** → **data isn't stationer**

<img width="472" height="276" alt="Screenshot 2026-02-16 at 21 11 51" src="https://github.com/user-attachments/assets/0e285e41-0cc5-48d7-b9fe-9abbbe6f6eb5" />

# 🔮 2. Forecasting Model (Notebook 3)

This notebook creates forecasting models using classical time-series algorithms.
You may have implemented ARIMA, SARIMA, Prophet, or other models.

## Key steps include:

1. Setting Order Date as index

2. Checking stationarity (ADF test)

3. Differencing if needed

4. Auto ARIMA / SARIMAX model creation

5. Model training & validation split

6. Forecast visualization

7. Evaluation (RMSE, MAPE, MAE)

# Create Sales Forecast Model

A. Seasonal Naive Bayes 
B. ARIMA
C. SARIMAX

<img width="952" height="828" alt="image" src="https://github.com/user-attachments/assets/0613eef8-f129-4b36-be2b-0dd4fa730e71" />

<img width="476" height="451" alt="Screenshot 2026-02-16 at 21 18 08" src="https://github.com/user-attachments/assets/b8a5edb9-b8ac-4076-9989-8dcb669beb11" />

<img width="655" height="566" alt="Screenshot 2026-02-16 at 21 19 00" src="https://github.com/user-attachments/assets/2a3f5a7e-2444-4272-8f29-fa3c26d2c13c" />


## Create Prediction for 2018

Disclaimer:
## The model that I picked should be seasonal naive bayes --> the error (RMSE/MAE) is smallest.

## But for learning purpose only I created Sales Forecasting using SARIMAXto predict Sales in 2018
   

<img width="655" height="534" alt="Screenshot 2026-02-16 at 21 20 35" src="https://github.com/user-attachments/assets/ea60ca09-be50-49c4-862c-509024bc5610" />
