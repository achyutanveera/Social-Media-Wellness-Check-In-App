# Mind the Gap: source code guide

Mind the Gap is a free, optional student wellness check-in. This repository holds all of its source code.

| File | What it is | Language |
|---|---|---|
| mind_the_gap_student_app.html | Student check-in app (questions, break questions, brain-break games, results page). Deployed as index.html on Firebase Hosting | HTML, CSS, JavaScript |
| mind_the_gap_counselor_dashboard.html | Counselor dashboard (student list, charts, trend and jump flags, model estimate). Deployed as counselor.html | HTML, CSS, JavaScript |
| chart.umd.js | Chart.js 4.4.4, a third-party charting library (MIT license), not written by me | JavaScript |
| firestore.rules | Database security rules (who can write and read) | Firebase rules language |
| set_counselor.js | One-time script that gives a Google account counselor access | JavaScript (Node.js) |
| preprocess.py | Cleans the Kaggle dataset and builds model inputs | Python |
| train_export.py | Trains the MLP models and saves them | Python (scikit-learn) |
| api.py | Prediction API served on Render | Python (FastAPI) |
| model_full.joblib, model_no_sum.joblib, meta.json | Trained models and their accuracy numbers | binary / JSON |
| requirements.txt | Python packages for the API | text |

Services: Firebase Hosting, Firestore and Authentication; Render (API hosting); GitHub (code).
Firebase JS SDK 10.13.0 is loaded from Google.
Dataset: Kaggle "social-media-and-mental-health" (souvikahmed071).
