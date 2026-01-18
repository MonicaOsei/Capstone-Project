# Movie Recommender System API

## Problem Description

With the rapid growth of digital content, users are often overwhelmed by the sheer number of movies available on streaming platforms. Finding movies that match individual preferences can be time-consuming and inefficient without intelligent filtering.

This project provides a **machine learning–based movie recommender system** that predicts whether a user will like a movie based on user and movie features. The system is designed to help platforms deliver **personalized movie recommendations**, improve user engagement, and enhance the overall viewing experience.

---

## Solution Overview

This project implements a **content-based / supervised learning recommender system** using classical machine learning algorithms such as:

* Logistic Regression
* Decision Tree
* Random Forest

The model is trained on a movie dataset containing user attributes and movie features (e.g. age, gender, genre preferences, ratings). The trained model is serialized and deployed using a **Flask API**, allowing external applications to request recommendations via HTTP requests.

---

## API Functionality

The API exposes the following routes:

* `/`
  Checks whether the API is running.

* `/predict`
  Accepts user and movie features in JSON format and returns a prediction indicating whether the user is likely to enjoy the movie.

---

## Installation & Running the Project

### Prerequisites

* Python 3.11 or higher
* Git
* Docker (optional, for containerized deployment)

---

### Steps to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/MonicaOsei/movie_recommender.git
cd movie_recommender

# 2. Create and activate virtual environment
python -m venv recommender_env
recommender_env\Scripts\activate      # Windows
source recommender_env/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
python app.py

# 5. Test the API using curl
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d "{\"features\": [25, 0, 3, 1]}"
```

---

### Running with Docker (Optional)

```bash
# 1. Build Docker image
docker build -t movie_recommender_app .

# 2. Run Docker container
docker run -p 8000:8000 movie_recommender_app

# 3. Test API
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d "{\"features\": [25, 0, 3, 1]}"
```

---

## Input Format

The `/predict` endpoint expects input in the following JSON format:

```json
{
  "features": [age, gender, genre_preference, rating]
}
```

> **Note:** Feature order and encoding must match the model’s training configuration.

---

## Output Format

The API returns a JSON response similar to:

```json
{
  "prediction": 1,
  "probability": 0.82
}
```

Where:

* `prediction = 1` → Recommended
* `prediction = 0` → Not recommended

---

## Project Structure

```
movie_recommender/
│
├── app.py                # Flask API
├── model.pkl             # Trained recommender model
├── requirements.txt      # Dependencies
├── Dockerfile            # Docker configuration
├── notebooks/            # EDA and model training notebooks
├── data/                 # Dataset
└── README.md             # Project documentation
```

---

## Importance & Applications

* Enables **personalized movie recommendations**
* Improves user engagement on streaming platforms
* Reduces information overload
* Can be integrated into web or mobile applications
* Demonstrates end-to-end ML deployment (training → API → Docker)





