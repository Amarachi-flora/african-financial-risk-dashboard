# 🌍 African Financial Customer Intelligence Platform  
### Customer Financial Risk Prediction & Sentiment Analysis System

**Domain:** African Financial Markets Analytics  
**Program:** [DataVerse Africa Internship Cohort 3.0](https://dataverseafrica.org/) — Data Analytics Track  
**Duration:** 12 Weeks (Training + Project-Based)  
**Team:** Amarachi Florence, Thato Maelane, Philip Odiachi, Mavis  

---

##  Project Overview

The **African Financial Customer Intelligence Platform** is an end-to-end analytics solution developed to help financial institutions in Africa better understand customer behavior, assess financial risk, and derive business insights that drive real operational improvements. The platform integrates machine learning, sentiment analysis, interactive web analytics, and executive reporting into a unified analytical ecosystem.

The system was validated using **5,200+ real customer records**, but it was designed with adaptive intelligence so that users can upload **any financial dataset** and obtain structured analytics and actionable insights.

---

## 🌐 Live Platforms & Resources

 **Streamlit Interactive App (Flexible Analytics):**  
[https://african-financial-risk-dashboard-d.streamlit.app/](https://african-financial-risk-dashboard-d.streamlit.app/)

 **Power BI Executive Dashboard Suite (Production Insights):**  
[https://bit.ly/45za0YJ](https://bit.ly/45za0YJ)

 **Project Report (PPT):**  
[https://docs.google.com/presentation/d/1JN4UVYeWbU-VgZt-d_Dew5ne6vzgucit/edit](https://docs.google.com/presentation/d/1JN4UVYeWbU-VgZt-d_Dew5ne6vzgucit/edit)

 **DataVerse Africa – Internship & Community:**  
[https://dataverseafrica.org/](https://dataverseafrica.org/) — A community and training hub empowering African data professionals with project-based learning, mentorship, and real business problem solving.

---

##  Core Capabilities

This platform provides:

- Customer segmentation (automated clustering)  
- Predictive risk scoring  
- Sentiment analysis on customer feedback  
- Digital channel adoption insights  
- Interactive visualization dashboards  
- Exportable analytic outputs  
- Business recommendations tied to insights

---

##  How to Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
 

📊 Power BI Executive Dashboard Suite

Below are summaries and insights from the six executive dashboards built on the fixed 5,200-record dataset:

🔹 Dashboard 1 — Customer Risk Overview

This dashboard gives a high-level view of the overall customer base, segmented by risk profiles and key financial indicators. A sizeable percentage of customers are classified as high risk, signaling areas where credit policies or engagement strategies could be improved.

Business Insight:
The high concentration of risk among certain segments suggests a need for tighter credit monitoring and customer education initiatives focused on financial health.

Recommendations:
Introduce tailored financial literacy programs and early warning systems that help customers manage credit and avoid defaults.

Dashboard Image:


🔹 Dashboard 2 — Customer Segments Analysis

This dashboard explores customer segmentation and how these groups differ across demographic, geographic, and behavioral patterns. It highlights urban centers like Lagos and Abuja as hotspots for high-value behavior.

Business Insight:
Customer behavior varies significantly across segments — some with high spending and digital transaction rates, others more traditional in engagement.

Recommendations:
Segment-driven strategies should focus on customizing product offerings and outreach tactics to match distinct group needs, especially in urban clusters.

Dashboard Image:


🔹 Dashboard 3 — Digital Transformation Insights

Focused on digital adoption, this dashboard tracks mobile app uptake, digital engagement metrics, and channel usage trends.

Business Insight:
A low digital adoption rate suggests that many customers still prefer traditional banking channels, limiting scalability and cost efficiency.

Recommendations:
Offer incentives for customers to use digital platforms — such as in-app perks, easier navigation, and targeted promotions — to accelerate digital transformation.

Dashboard Image:


🔹 Dashboard 4 — Financial Health Metrics

This visualization describes the distribution of credit scores and the total value of the loan portfolio. It provides deep insight into lending performance and customer credit health.

Business Insight:
Many customers fall into ‘Poor’ or ‘Fair’ credit score categories, directly influencing risk levels and loan performance outcomes.

Recommendations:
Launch credit-improvement programs and tailored lending products that help customers progressively build better credit profiles.

Dashboard Image:


🔹 Dashboard 5 — Voice of Customer (Sentiment Analysis)

This dashboard analyzes customer feedback sentiment using VADER sentiment scoring and visualizes sentiment trends across feedback categories.

Business Insight:
Negative sentiment is prevalent around service reliability and technical issues, indicating potential gaps in customer experience.

Recommendations:
Deploy structured feedback response processes and invest in support training to resolve issues promptly, thereby improving sentiment and retention.

Dashboard Image:


🔹 Dashboard 6 — Prediction Risk Monitoring

Here, the focus is on identifying individual customers with high default probability or risk signals, based on predictive models.

Business Insight:
Identifying at-risk customers ahead of time enables preemptive interventions that reduce losses and improve recovery prospects.

Recommendations:
Implement personalized outreach and risk mitigation packages for high-risk segments, supported by monitoring dashboards that update in near real-time.

Dashboard Image:


 Streamlit App – Features

The Streamlit application accepts any financial dataset and automatically:

Detects key columns (metadata and types)

Performs intelligent cleaning and imputation

Runs segmentation & risk prediction

Executes sentiment analysis on feedback text

Generates interactive charts & summaries

Produces downloadable reports

Supported input formats:

CSV

Excel

This flexible analytical tool empowers users to upload their own real-world datasets and obtain actionable insights in minutes.

🛠 Installation & Setup

Prerequisites:

Python 3.8+

pip package manager

Local Setup:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
python -m streamlit run streamlit_app.py

Testing:

test_vader.py – VADER sentiment test

test_vscode.py – Environment sanity test

 API Endpoints
Endpoint	Method	Description
/health	GET	System health check
/predict	POST	Predict risk for a single customer
/predict/batch	POST	Batch predictions
/clusters	GET	Get cluster definitions
/demo	GET	Demo prediction

 Report & Presentation

Capstone Project Report:
https://docs.google.com/presentation/d/1JN4UVYeWbU-VgZt-d_Dew5ne6vzgucit/edit

🎓 Acknowledgments

This work was completed as part of the DataVerse Africa Internship Program. Special thanks to the mentors, program partners, and the open-source community.

Learn more at:

🌐 DataVerse Africa


 License

Licensed under the MIT License. See the LICENSE file for details.