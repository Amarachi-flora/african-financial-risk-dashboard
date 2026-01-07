# Customer Financial Risk Prediction Dashboard

##  Project Overview

A comprehensive machine learning pipeline and interactive dashboard for predicting customer financial risk in African markets. This project analyzes **5,200+ customer records** to provide actionable insights, customer segmentation, and risk predictions through an intuitive **Streamlit** interface.

---

##  Key Features

###  Data Analysis & ML Pipeline

* **5,200+ Customer Records Analysis**: Comprehensive financial data processing
* **Customer Segmentation**: Unsupervised clustering into distinct customer groups
* **Risk Assessment**: Predictive modeling for financial risk scoring
* **Sentiment Analysis**: NLP-powered analysis of customer feedback
* **Feature Engineering**: Advanced feature creation for better predictions

###  Interactive Dashboard

* **Real-time Predictions**: API-driven single and batch predictions
* **Customer Analytics**: Deep filtering and segmentation analysis
* **Visual Analytics**: Interactive charts and business intelligence
* **Power BI Integration**: Ready-to-use data exports for BI dashboards
* **Team Collaboration**: Project documentation and team information

---

##  Quick Start

###  Prerequisites

* Python 3.8+
* pip package manager

###  Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/customer-risk-dashboard.git
cd customer-risk-dashboard
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up directories**

```bash
mkdir -p outputs models charts powerbi eda_reports api
```

---

## ▶️ Running the Application

### 1️⃣ Start the ML Pipeline

```bash
python project_main.py
```

Processes the data, trains models, and generates outputs.

### 2️⃣ Start the API Server

```bash
python api/api_main.py
```

### 3️⃣ Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

### 🌐 Access Points

* **Dashboard**: [http://localhost:8501](http://localhost:8501)
* **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

##  Project Structure

```text
customer-risk-dashboard/
│
├── streamlit_app.py
├── project_main.py
├── requirements.txt
│
├── api/
│   └── api_main.py
│
├── outputs/
│   ├── processed_data.csv
│   ├── cluster_profiles.csv
│   ├── business_recommendations.csv
│   └── final_project_report.txt
│
├── models/
│   ├── scaler.pkl
│   ├── pca_model.pkl
│   └── kmeans_model.pkl
│
├── charts/
├── powerbi/
├── eda_reports/
│
└── README.md
```

---

##  Technical Stack

###  Backend

* **Python**
* **Scikit-learn**
* **Pandas & NumPy**
* **FastAPI**
* **Joblib**

###  Frontend

* **Streamlit**
* **Plotly**
* **Matplotlib & Seaborn**

###  Data Processing

* Feature Engineering
* PCA (Dimensionality Reduction)
* K-Means Clustering
* NLP Sentiment Analysis

---

## 📈 Key Deliverables

### 1️⃣ Customer Segmentation

* 6 distinct customer clusters
* Detailed segment profiles
* Targeted business strategies

### 2️⃣ Risk Prediction

* Real-time risk scoring
* Low / Medium / High risk classification
* Digital adoption analysis

### 3️⃣ Business Intelligence

* Power BI–ready datasets
* Actionable recommendations
* Financial KPIs

### 4️⃣ Interactive Dashboard

* 7 multi-page views
* Real-time filtering
* Exportable reports

---

##  Dashboard Pages

*  **Dashboard** – Executive summary
*  **Customer Analysis** – Deep filtering
*  **Clusters** – Segment comparisons
*  **Predict** – Risk prediction & API testing
*  **Insights** – Business recommendations
*  **Team** – Project contributors & [Dataverse Africa](https://dataverseafrica.org)
*  **Settings** – System configuration

---

##  Sample Outputs

### Customer Segments

* Digital-First High Spenders – 22%
* Traditional Low-Risk – 18%
* High-Risk Low Spenders – 15%
* Positive High Spenders – 20%
* Medium Digital Medium Risk – 25%

### Key Metrics

* Total Customers: **5,200**
* Avg Credit Score: **645**
* Avg Monthly Spend: **₦150,000**
* Digital Adoption: **68%**
* High-Risk Customers: **12.5%**

---

##  API Endpoints

| Endpoint       | Method | Description       |
| -------------- | ------ | ----------------- |
| /health        | GET    | API health check  |
| /predict       | POST   | Single prediction |
| /predict/batch | POST   | Batch predictions |
| /clusters      | GET    | Cluster info      |
| /demo          | GET    | Demo prediction   |

---

##  Requirements

See `requirements.txt`. Key packages:

* streamlit==1.28.0
* pandas==2.1.3
* plotly==5.17.0
* scikit-learn==1.3.2
* fastapi==0.104.1
* uvicorn==0.24.0

---

## 🤝 Team

**Dataverse Africa Internship Program**

* **Amarachi Florence** – Financial Data and MEAL Analyst
* Thato Maelane – Data Scientist 
* Philip Odiachi – Data Analyst
* Mavis – Business Analyst

---

##  About Dataverse Africa

[Dataverse Africa](https://dataverseafrica.org) is empowering Africa’s digital future through:

* Training next-generation data scientists
* Applied AI research for African markets
* Industry collaboration
* Data-driven solutions for real-world problems

---

##  Business Impact

### For Financial Institutions

* Improved risk assessment
* Better customer segmentation
* Digital channel optimization
* Enhanced customer sentiment insights

### For Customers

* Personalized financial products
* Improved risk management
* Better digital banking experience

---

##  Getting Started with Your Data

1. Prepare your dataset
2. Run `python project_main.py`
3. Explore insights via Streamlit
4. Apply business recommendations

---

##  License

Developed as part of the **Dataverse Africa Internship Program**. All rights reserved.

---

##  Links

* **Dashboard**: [http://localhost:8501](http://localhost:8501)
* **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Dataverse Africa**: [https://dataverseafrica.org](https://dataverseafrica.org)

---

**Built with ❤️ for African Financial Markets — Empowering Data-Driven Decisions**
