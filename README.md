# 🚀 Remote Worker Productivity Prediction

This project predicts the **productivity score of remote workers** based on work behavior, time management, and tool usage metrics. It demonstrates a complete **end-to-end machine learning workflow**, including data preprocessing, model training, and deployment as a web service using Docker.

---

## 📌 Problem Description

With the rapid growth of remote work, organizations need effective ways to understand and improve employee productivity.

The objective of this project is to **predict a continuous productivity score (`productivity_score`)** using features such as work location, industry sector, working hours, break frequency, task completion behavior, tool usage, and focus time. This task is formulated as a **supervised regression problem**.

---

## 📊 Dataset

The dataset is sourced from Kaggle: **Remote Worker Productivity Dataset**.

Target variable:
- `productivity_score`

Excluded columns:
- `worker_id` (identifier only)
- `productivity_label` (derived from target, removed to avoid target leakage)

---

## 🧠 Modeling Approach

Raw data is cleaned directly inside `train.py`. Rows with missing values are removed, and categorical variables are encoded using **DictVectorizer**. The vectorizer is fitted only on the training set to prevent data leakage and reused during validation, testing, and inference.

Models trained and compared using validation RMSE:
- Linear Regression (baseline)
- Ridge Regression (regularized baseline)
- Random Forest Regressor (final model)

The final model is evaluated once on a held-out test set.

---

## 📁 Project Structure

├── README.md  
├── notebook.ipynb  
├── train.py  
├── predict.py  
├── model.bin  
├── requirements.txt  
├── Dockerfile  
└── data/  
&nbsp;&nbsp;&nbsp;&nbsp;└── remote_worker_productivity_1000.csv  

---

## ⚙️ How to Run the Project

### Train the Model

python train.py

This script loads the raw dataset, performs data cleaning and feature encoding, trains multiple models, evaluates them, and saves the final model and DictVectorizer to `model.bin`.

---

### Run the Prediction Service Locally

python predict.py

The API will be available at:

http://localhost:9696

---

### Make a Prediction Request

```
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{  
  "location_type": "City",  
  "industry_sector": "Healthcare",  
  "age": 27,  
  "experience_years": 5,  
  "average_daily_work_hours": 8.2,  
  "break_frequency_per_day": 3,  
  "task_completion_rate": 85.0,  
  "late_task_ratio": 0.15,  
  "calendar_scheduled_usage": 60.0,  
  "focus_time_minutes": 140,  
  "tool_usage_frequency": 10,  
  "automated_task_count": 5,  
  "AI_assisted_planning": 1,  
  "real_time_feedback_score": 80  
}'
```
Example response:

{  
  "predicted_productivity_score": 36.8  
}

---

## 🐳 Docker Deployment

```docker build -t remote-productivity .```

```docker run -p 9696:9696 remote-productivity```

The service will be available at:

http://localhost:9696/predict

---


## 🛠 Technologies Used

Python  
pandas  
numpy  
scikit-learn  
Flask  
DictVectorizer  
Docker  
gunicorn  

---




