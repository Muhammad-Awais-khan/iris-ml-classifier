# 🌸 Iris Flower Classifier Web App

This is a web application that uses a trained K-Nearest Neighbors (KNN) model to classify iris flowers into three species:
- Setosa
- Versicolor
- Virginica

---

## 🧠 Tech Stack
- **Frontend:** HTML, CSS (with Flask templates)
- **Backend:** Flask (Python)
- **Machine Learning:** scikit-learn (KNN Classifier)
- **Deployment:** Render

---

## 📦 Project Structure

```
iris-ml-classifier/
├── app.py                  # Flask web server
├── train_model.py          # Script to train and save the KNN model
├── knn_model.joblib        # Saved ML model
├── templates/
│   ├── index.html          # Input form
│   └── result.html         # Prediction result display
├── static/
│   └── style.css           # CSS styling
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment config for Render
└── README.md               # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Muhammad-Awais-khan/iris-ml-classifier
   cd iris-webapp
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

5. Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 🌐 Live Demo

[https://youtu.be/be9qfSySxJw](https://youtu.be/be9qfSySxJw)

---

## 📁 To Train the Model

If you want to retrain the model or tweak parameters:

```bash
python train_model.py
```

---

## 🙌 Credits

Created by **Muhammad Awais**  
Final Project for Harvard's **CS50x** course.

---
