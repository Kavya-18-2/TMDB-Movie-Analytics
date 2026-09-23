# 🎬 TMDB Movie Analytics

## 📌 Project Overview

TMDB Movie Analytics is a Python-based data analytics and machine learning project that analyzes movie data from The Movie Database (TMDB).

The project combines movie information and credits data to explore movie trends, ratings, popularity, revenue, genres, and other movie-related characteristics.

A Machine Learning model is also developed to predict the expected movie rating based on selected movie features.

The project provides an interactive Streamlit web application for data visualization, movie searching, and rating prediction.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze TMDB movie datasets.
* Combine movie and credits datasets.
* Clean and preprocess movie data.
* Perform Exploratory Data Analysis (EDA).
* Identify movie trends and patterns.
* Analyze movie ratings, revenue, popularity, and genres.
* Visualize important insights using charts.
* Build a Machine Learning model for rating prediction.
* Develop an interactive Streamlit application.

---

## 📂 Datasets

The project uses two TMDB datasets:

### 1. TMDB Movies Dataset

Contains information such as:

* Movie title
* Budget
* Revenue
* Popularity
* Release date
* Runtime
* Vote average
* Vote count
* Genres
* Keywords
* Production information

### 2. TMDB Credits Dataset

Contains:

* Movie ID
* Movie title
* Cast
* Crew

The datasets are combined using the common movie title field.
## Dataset Source

Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook
* Streamlit

---

## 🤖 Machine Learning

A Random Forest Regression model is used for movie rating prediction.

### Features used

* Budget
* Revenue
* Popularity
* Runtime
* Vote Count
* Release Year

### Target

* Vote Average

The trained model is saved as:

```text
model/movie_model.pkl
```

---

## 📊 Exploratory Data Analysis

The project analyzes:

* Number of movies released by year
* Top movie genres
* Movie rating distribution
* Top movies by revenue
* Relationship between budget and revenue
* Relationship between popularity and rating
* Correlation between numerical movie features

The generated visualizations are stored in the:

```text
images/
```

folder.

---

## 🌐 Streamlit Application

The Streamlit application provides three main sections:

### 📊 Dashboard

Displays:

* Total number of movies
* Average movie rating
* Average popularity
* Total revenue
* Movies released by year
* Rating distribution

### 🔎 Movie Search

Users can search for a movie by title and view information such as:

* Release date
* Rating
* Vote count
* Popularity
* Revenue

### 🤖 Rating Prediction

Users can enter movie information and the trained Machine Learning model predicts the expected movie rating.

---

## 📁 Project Structure

```text
TMDB-Movie-Analytics/
│
├── data/
│   ├── tmdb_movies.csv
│   └── tmdb_credits.csv
│
├── images/
│   ├── movies_by_year.png
│   ├── top_genres.png
│   ├── rating_distribution.png
│   ├── top_revenue.png
│   ├── budget_vs_revenue.png
│   ├── popularity_vs_rating.png
│   └── correlation_heatmap.png
│
├── model/
│   └── movie_model.pkl
│
├── notebooks/
│   └── TMDB_Movie_Analytics.ipynb
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Step 1: Clone or download the project

Open the project folder in PowerShell.

### Step 2: Create a virtual environment

```powershell
python -m venv venv
```

### Step 3: Activate the virtual environment

```powershell
venv\Scripts\activate
```

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Machine Learning model

Run:

```powershell
python train_model.py
```

This creates:

```text
model/movie_model.pkl
```

### Start the Streamlit application

Run:

```powershell
streamlit run app.py
```

The Streamlit application will open in your web browser.

---

## 📓 Jupyter Notebook

The complete data analysis and visualization process is available in:

```text
notebooks/TMDB_Movie_Analytics.ipynb
```

The notebook contains:

1. Data loading
2. Dataset inspection
3. Dataset merging
4. Data cleaning
5. Missing-value handling
6. Exploratory Data Analysis
7. Data visualization
8. Correlation analysis

---

## 📈 Model Evaluation

The Random Forest model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The evaluation results are displayed when `train_model.py` is executed.

---

## 🔐 Responsible AI Considerations

### Fairness

The model predictions depend on the available historical movie data. The model should not be treated as a guarantee of a movie's actual future rating.

### Transparency

The prediction is based on the input features used during model training, including budget, revenue, popularity, runtime, vote count, and release year.

### Ethics

The system is designed for movie analytics and educational purposes. Predictions should not be used to make harmful or misleading claims about movies, filmmakers, actors, or audiences.

### Privacy

The project uses movie-related datasets and does not intentionally collect or process private personal information.

---

## 🚀 Future Improvements

Future versions could include:

* Movie recommendation system
* Advanced Natural Language Processing
* Sentiment analysis of movie reviews
* Genre prediction
* Actor and director analytics
* Interactive Plotly visualizations
* More advanced Machine Learning models
* Deployment as a web application
* Real-time TMDB API integration

---

## 📌 Project Purpose

This project was developed for educational and data science purposes to demonstrate:

* Data preprocessing
* Exploratory Data Analysis
* Data visualization
* Machine Learning
* Model evaluation
* Python programming
* Streamlit application development

---

## 👩‍💻 Author

**Yernenti Kavya**

B.Tech – Computer Science Engineering and Data Science

Raghu Engineering College, Vizag
