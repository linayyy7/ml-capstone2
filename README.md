🚀 Remote Worker Productivity Prediction Service

This repository contains a complete machine learning project that predicts the task completion rate of remote workers 📈 based on their work behavior, experience, and scheduling patterns. The project demonstrates an end-to-end machine learning workflow, including data analysis, model training, evaluation, and deployment as a REST API using Docker 🐳.
This project is developed as part of the ML Zoomcamp program 🎓.

🌍 Background and Motivation

With the increasing adoption of remote and hybrid work models, organizations face challenges in understanding employee productivity without direct supervision 👀. Traditional productivity metrics are often subjective or difficult to measure consistently.
This project explores whether observable behavioral signals—such as daily working hours, break frequency, experience level, and calendar usage—can be used to quantitatively predict task completion performance for remote workers using supervised machine learning 🤖.

📊 Dataset Description

Each record in the dataset represents one remote worker.
The input features used in this project include location_type (worker’s location category), industry_sector (industry of employment), age, experience_years, average_daily_work_hours, break_frequency_per_day, calendar_scheduled_usage, and late_task_ratio.
The target variable is task_completion_rate, which represents the proportion of tasks completed successfully.

The column worker_id is used only as an identifier and is removed during preprocessing.
The dataset file should be placed at data/remote_worker_productivity.csv.

🗂️ Project Structure

The repository is organized as follows: README.md for documentation, notebook.ipynb for exploratory data analysis, train.py for model training, predict.py for serving predictions, model.bin for the trained model artifact, requirements.txt for dependencies, Dockerfile for containerization, and a data folder containing the dataset CSV file.

⚙️ Machine Learning Workflow

The project follows a standard and reproducible machine learning workflow.
First, exploratory data analysis (EDA) is performed to understand the distribution of the target variable, analyze feature–target relationships, and identify potential data quality issues such as outliers.
Next, preprocessing and feature engineering are applied. Categorical features are encoded using one-hot encoding, while numerical features are passed directly. Preprocessing and modeling are combined into a single pipeline to ensure consistency between training and inference.
For modeling, a baseline linear regression model is trained for comparison, followed by a Random Forest Regressor 🌲 as the final model. Model performance is evaluated using RMSE (Root Mean Squared Error), and the best-performing model is selected.
Finally, the trained pipeline, including preprocessing steps, is serialized and saved as model.bin 💾.

🏋️ Model Training

To train the model locally, the training script loads the dataset, splits the data into training and test sets, trains the model, evaluates its performance on the test data, and saves the trained model artifact to disk.
Training is performed using the train.py script with the dataset path and output model path provided as arguments.

🌐 Prediction API

A lightweight REST API is implemented using Flask 🧪 to serve predictions.
When the service is started, it listens on port 9696 🔌 and exposes two endpoints: a health check endpoint (GET /health) and a prediction endpoint (POST /predict).
The prediction endpoint accepts worker attributes as JSON input and returns a predicted task completion rate.

A typical request includes fields such as location_type, industry_sector, age, experience_years, average_daily_work_hours, break_frequency_per_day, calendar_scheduled_usage, and late_task_ratio.
The response contains a single value: predicted_task_completion_rate.

🐳 Docker Deployment

The prediction service can be packaged and deployed locally using Docker.
A Docker image is built using the provided Dockerfile, and the container exposes port 9696 so the API can be accessed at http://localhost:9696
 once running.

📦 Dependencies

All required Python dependencies are listed in requirements.txt.
The main libraries used in this project include pandas, numpy, scikit-learn, Flask, and gunicorn.

✅ ML Zoomcamp Deliverables

This project fulfills all ML Zoomcamp requirements, including a clear problem definition, dataset documentation, exploratory data analysis, feature engineering, multiple models and evaluation, reproducible training scripts, model serialization, a prediction web service, and Dockerized deployment.

🔮 Limitations and Future Work

Future improvements may include experimenting with gradient boosting models, adding model explainability techniques such as SHAP 📊, performing subgroup error analysis, and deploying the service to a cloud platform ☁️.

