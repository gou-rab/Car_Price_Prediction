# 🚗 Car Price Predictor

A Machine Learning web application that predicts the **selling price of a used car** based on real listings from the Quikr dataset. Built with **Linear Regression**, **Flask**, and a custom dark-themed HTML frontend.

---

## 📸 Preview

![Car Price Predictor UI](eda_plots.png)

---

## 📁 Project Structure

```
car-price-predictor/
│
├── quikr_car - quikr_car.csv       # Raw dataset (892 listings)
├── car_price_predictor.py          # Data cleaning, EDA, model training
├── app.py                          # Flask backend API server
├── index.html                      # Frontend web UI
├── requirements.txt                # Python dependencies
│
│   ── Generated after training ──
├── LinearRegressionModel.pkl       # Trained ML model
├── feature_columns.pkl             # One-hot encoded feature columns
├── eda_plots.png                   # EDA visualization
├── company_price.png               # Avg price by company chart
└── actual_vs_predicted.png         # Model evaluation plot
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Quikr](https://www.quikr.com) used car listings |
| Raw rows | 892 |
| Rows after cleaning | 815 |
| Features | `name`, `company`, `year`, `kms_driven`, `fuel_type` |
| Target | `Price` (INR ₹) |

### Data Cleaning Steps
- Removed rows where `year` was non-numeric
- Removed rows where `Price` was `"Ask For Price"`
- Stripped commas from price strings and converted to integer
- Removed `" kms"` suffix from `kms_driven` and converted to integer
- Dropped rows with null `fuel_type`
- Truncated car name to first 3 words to reduce noise
- Removed price outliers above ₹60,00,000
- Removed kms outliers above 5,00,000

---

## 🤖 Machine Learning

| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Library | scikit-learn |
| Encoding | One-Hot Encoding (`pd.get_dummies`, `drop_first=True`) |
| Features after encoding | 281 |
| Train/Test Split | 80% / 20% |
| Best R² Score | **0.87** |
| Selection strategy | Best model across 1000 random train/test splits |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn Linear Regression |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Backend API | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Model Persistence | pickle |

---

## ⚙️ Setup & Installation

### 1. Clone or download the project

```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Run this **once** to clean the data, train the model, and save the `.pkl` files:

```bash
python car_price_predictor.py
```

You will see output like:
```
📁 Working directory: /your/project/folder
📄 Loading CSV: quikr_car - quikr_car.csv
✅ Cleaned shape: (815, 6)
⏳ Finding best model across 1000 random splits (please wait)...
   ... 0/1000 splits tested
   ... 200/1000 splits tested
   ...
🏆 Best R² Score across 1000 random splits: 0.8716
✅ Model saved → LinearRegressionModel.pkl
✅ Features saved → feature_columns.pkl
```

> ⚠️ This step takes 1–2 minutes due to the 1000 split iterations. This is expected — do not close the terminal.

### 4. Start the Flask server

```bash
python app.py
```

Server will start at: `http://127.0.0.1:5000`

### 5. Open the web app

Open `index.html` in your browser. The status badge in the top-right will turn **green** when the Flask server is detected.

---

## 🔗 How It All Connects

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  quikr_car.csv                                          │
│       │                                                 │
│       ▼                                                 │
│  car_price_predictor.py  ──trains──▶  .pkl files        │
│                                           │             │
│                                           ▼             │
│  index.html  ──POST /predict──▶  app.py (Flask)         │
│      ▲                               │                  │
│      │                               ▼                  │
│      └──────── JSON price ◀── LinearRegressionModel.pkl │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

1. **`car_price_predictor.py`** reads the CSV, cleans data, trains the model, saves `.pkl`
2. **`app.py`** loads the `.pkl` on startup, exposes a `/predict` POST endpoint
3. **`index.html`** collects user input, sends it to Flask, displays the predicted price

---

## 🌐 API Reference

### `POST /predict`

Predicts the price of a car given its details.

**Request Body (JSON):**
```json
{
  "name":       "Honda City",
  "company":    "Honda",
  "year":       2016,
  "kms_driven": 40000,
  "fuel_type":  "Petrol"
}
```

**Response:**
```json
{
  "success": true,
  "price": 425000.0,
  "price_formatted": "₹4,25,000",
  "input": {
    "name": "Honda City",
    "company": "Honda",
    "year": 2016,
    "kms_driven": 40000,
    "fuel_type": "Petrol"
  }
}
```

### `GET /model-info`

Returns information about the loaded model.

**Response:**
```json
{
  "model_type": "LinearRegression",
  "total_features": 281,
  "feature_names": ["year", "kms_driven", "name_Audi A4 1.8", "..."]
}
```

---

## 📈 Model Insights

- **Newer cars** → Higher predicted price (year has a positive coefficient)
- **More kms driven** → Lower predicted price
- **Diesel** cars tend to be priced higher than Petrol for the same model
- **Luxury brands** (Audi, BMW, Mercedes) have significantly higher base prices
- **Budget hatchbacks** (Alto, Nano, Spark) cluster in the ₹50,000–₹3,00,000 range

---

## 🚀 Supported Car Companies

Audi, BMW, Chevrolet, Datsun, Fiat, Force, Ford, Hindustan, Honda, Hyundai, Jaguar, Jeep, Land Rover, Mahindra, Maruti Suzuki, Mercedes-Benz, Mini, Mitsubishi, Nissan, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo

---

## ⚠️ Known Limitations

- Model is trained only on **Quikr listings** — predictions reflect used car market prices from that platform
- Car names not present in the training data will be treated as unknown (encoded as all zeros)
- Prices above ₹60 lakh and kms above 5 lakh were excluded as outliers during training
- The model may not generalize well to very rare or luxury-only models with few training samples

---

## 🙌 Acknowledgements

- Dataset sourced from **Quikr** used car listings
- ML pipeline inspired by standard regression tutorials for the Quikr car dataset
- Built with [scikit-learn](https://scikit-learn.org/), [Flask](https://flask.palletsprojects.com/), and [pandas](https://pandas.pydata.org/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
