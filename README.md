# Week-2: Telecom Data Analysis and User Insights

## Project Overview

This project involves analyzing telecom data from TellCo, a mobile service provider, to uncover insights into customer behavior, user engagement, and satisfaction. The analysis will guide strategic decisions for potential investment or operational improvements.

### Key Objectives:
1. Perform User Overview Analysis:
   - Identify the most commonly used handsets and manufacturers.
   - Aggregate and analyze user behavior metrics for applications like YouTube, Social Media, and more.
2. Analyze User Engagement:
   - Measure session frequency, duration, and total traffic.
   - Cluster users based on engagement metrics using k-means.
3. Provide actionable insights to improve user experience and engagement.

---

## Folder Structure

```plaintext
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       ├── unittests.yml
├── scripts/
│   ├── __init__.py
│   ├── user_overview.py
│   └── user_engagement.py
├── notebooks/
│   ├── eda_user_overview.ipynb
│   └── eda_user_engagement.ipynb
├── tests/
│   ├── test_user_overview.py
│   └── test_user_engagement.py
├── data/
│   ├── telecom.sql
│   ├── telecom_schema.png
│   └── sample_data.csv
├── requirements.txt
├── Dockerfile
├── README.md
└── main.py
