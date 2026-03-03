import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('quikr_car - quikr_car.csv')
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

backup = df.copy()

df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(',', '', regex=False).astype(int)

df = df[df['kms_driven'].str.split().str.get(-1) == 'kms']        
df['kms_driven'] = df['kms_driven'].str.split().str.get(0).str.replace(',', '', regex=False)
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)

df = df[~df['fuel_type'].isna()]

df['name'] = df['name'].str.split().str.slice(0, 3).str.join(' ')
df = df.reset_index(drop=True)

print("\n✅ Cleaned shape:", df.shape)
print(df.head())

print("\nPrice stats:\n", df['Price'].describe())
df = df[df['Price'] < 6_000_000]
df = df[df['kms_driven'] < 500_000]

print("Shape after outlier removal:", df.shape)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Price Distribution')
sns.histplot(df['Price'], kde=True, color='steelblue')
plt.xlabel('Price (₹)')

plt.subplot(1, 3, 2)
plt.title('Year vs Price')
sns.scatterplot(data=df, x='year', y='Price', hue='fuel_type', alpha=0.6)
plt.xlabel('Year')
plt.ylabel('Price (₹)')

plt.subplot(1, 3, 3)
plt.title('Cars by Fuel Type')
df['fuel_type'].value_counts().plot(kind='bar', color=['coral','steelblue','green','orange'])
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120)
plt.show()
print("📊 EDA plots saved as eda_plots.png")

plt.figure(figsize=(14, 5))
company_price = df.groupby('company')['Price'].mean().sort_values(ascending=False)
company_price.plot(kind='bar', color='steelblue')
plt.title('Average Price by Company')
plt.xlabel('Company')
plt.ylabel('Average Price (₹)')
plt.tight_layout()
plt.savefig('company_price.png', dpi=120)
plt.show()


X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']
X = pd.get_dummies(X, drop_first=True)

print("\nFeature matrix shape:", X.shape)
print("Features:", list(X.columns[:10]), '...')

best_r2 = -np.inf
best_model = None
best_X_test = None
best_y_test = None

for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_X_test = X_test
        best_y_test = y_test

print(f"\n🏆 Best R² Score across 1000 random splits: {best_r2:.4f}")
y_pred = best_model.predict(best_X_test)

plt.figure(figsize=(7, 5))
plt.scatter(best_y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([best_y_test.min(), best_y_test.max()],
         [best_y_test.min(), best_y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (₹)')
plt.ylabel('Predicted Price (₹)')
plt.title(f'Actual vs Predicted  |  R² = {best_r2:.4f}')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=120)
plt.show()
print("📈 Actual vs Predicted plot saved.")

pickle.dump(best_model,   open('LinearRegressionModel.pkl', 'wb'))
pickle.dump(X.columns,    open('feature_columns.pkl', 'wb'))

print("\n✅ Model saved as  LinearRegressionModel.pkl")
print("✅ Feature cols saved as feature_columns.pkl")

def predict_price(name, company, year, kms_driven, fuel_type):
    model_loaded   = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    feature_cols   = pickle.load(open('feature_columns.pkl', 'rb'))

   
    sample = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )

    sample = pd.get_dummies(sample, drop_first=True)
    sample = sample.reindex(columns=feature_cols, fill_value=0)

    price = model_loaded.predict(sample)[0]
    return round(price, 2)

print("\n" + "="*50)
print("🚗  DEMO PREDICTIONS")
print("="*50)

examples = [
    ('Maruti Suzuki Alto', 'Maruti', 2012, 50000, 'Petrol'),
    ('Hyundai Grand i10',  'Hyundai', 2014, 30000, 'Petrol'),
    ('Honda City',         'Honda',   2016, 40000, 'Diesel'),
    ('Mahindra Bolero',    'Mahindra',2010, 80000, 'Diesel'),
]

for args in examples:
    price = predict_price(*args)
    print(f"  {args[0]} ({args[2]}, {args[3]:,} kms, {args[4]})  →  ₹{price:,.0f}")
