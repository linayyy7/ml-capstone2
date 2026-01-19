# 🚀 **Remote Worker Productivity Prediction Service**

This repository contains a complete machine learning project that predicts the task completion rate of remote workers 📈 based on their work behavior, experience, and scheduling patterns. The project demonstrates an end-to-end machine learning workflow, including data analysis, model training, evaluation, and deployment as a REST API using Docker 🐳. This project is developed as part of the ML Zoomcamp program 🎓.

## 🌍 **Background and Motivation**

With the increasing adoption of remote and hybrid work models, organizations face challenges in understanding employee productivity without direct supervision 👀. Traditional productivity metrics are often subjective or difficult to measure consistently.

This project investigates whether observable behavioral signals—such as daily working hours, break frequency, experience level, and calendar usage—can be used to quantitatively predict task completion performance for remote workers using supervised machine learning 🤖.

## 📊 **Dataset Description**

Each record in the dataset represents one remote worker along with productivity-related metrics.

The input features used in this project include:

**location_type** — worker's location category (e.g., Urban, Suburban, Rural)  
**industry_sector** — industry of employment  
**age** — worker age  
**experience_years** — total years of professional experience  
**average_daily_work_hours** — average number of hours worked per day  
**break_frequency_per_day** — number of breaks taken per day  
**calendar_scheduled_usage** — fraction of work time scheduled via a calendar  
**late_task_ratio** — proportion of tasks completed after their deadline  

The target variable is **task_completion_rate**, which represents the proportion of assigned tasks completed successfully.

The column **worker_id** is used only as an identifier and is removed during preprocessing.

The dataset file should be placed at:  
`data/remote_worker_productivity.csv`

## 🗂️ **Project Structure**

The repository is organized as follows:
## Project Structure

```text
├── README.md                  # Project documentation (this file)
├── notebook.ipynb             # Exploratory data analysis
├── train.py                   # Model training script
├── predict.py                 # Prediction API
├── model.bin                  # Trained model artifact
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── data/
    └── remote_worker_productivity.csv
```



## ⚙️ **Machine Learning Workflow**

The project follows a standard and reproducible machine learning workflow 🔁.

**1. Exploratory Data Analysis (EDA)**  
Performed to understand the distribution of the target variable, analyze relationships between features and productivity, and identify potential data quality issues such as outliers.

**2. Preprocessing and Feature Engineering**  
Categorical features are encoded using one-hot encoding, while numerical features are passed directly. Preprocessing and modeling are combined into a single pipeline to ensure consistency between training and inference.

**3. Modeling**  
A baseline linear regression model is trained for comparison, followed by a **Random Forest Regressor 🌲** as the final model. Model performance is evaluated using RMSE (Root Mean Squared Error), and the best-performing model is selected.

**4. Model Serialization**  
The trained pipeline—including preprocessing steps—is serialized and saved as `model.bin` 💾.

## 🏋️ **Model Training**

Model training is performed using the `train.py` script.  
The script loads the dataset, splits the data into training and test sets, trains the model, evaluates its performance on the test data, and saves the trained model artifact to disk.

## 🌐 **Prediction API**

A lightweight REST API is implemented using Flask 🧪 to serve predictions.

When the service is started, it listens on port `9696` 🔌 and exposes two endpoints:

**GET /health** — returns a health check response  
**POST /predict** — returns a predicted task completion rate  

The prediction endpoint accepts worker attributes as JSON input and returns a single value: `predicted_task_completion_rate`.

## 🐳 **Docker Deployment**

The prediction service can be packaged and deployed locally using Docker 🐳.

A Docker image is built using the provided `Dockerfile`, and the container exposes port `9696`, allowing the API to be accessed at:  
`http://localhost:9696`

## 📦 **Dependencies**

All required Python dependencies are listed in `requirements.txt`.  
The main libraries used in this project include `pandas`, `numpy`, `scikit-learn`, `Flask`, and `gunicorn`.

## ✅ **ML Zoomcamp Deliverables**

This project fulfills all ML Zoomcamp requirements, including:

- Clear problem definition
- Dataset documentation and usage instructions
- Exploratory data analysis
- Feature engineering and model selection
- Reproducible training scripts
- Model serialization
- Prediction web service
- Dockerized deployment

## 🔮 **Limitations and Future Work**

Future improvements may include:

- Experimenting with gradient boosting models
- Adding model explainability techniques such as SHAP 📊
- Performing subgroup error analysis
- Deploying the service to a cloud platform ☁️

