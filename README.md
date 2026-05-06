# 📩 Spam Detection with XLM-RoBERTa and Sentiment Analysis

## 📌 Project Description

This project implements a spam detection system using the **XLM-RoBERTa** transformer model. It classifies SMS messages as **Spam** or **Ham (Not Spam)** and also performs **Sentiment Analysis** to identify whether the message is positive, negative, or neutral.

---

## 🎯 Objectives

* Build an accurate spam detection model
* Use transformer-based NLP (XLM-RoBERTa)
* Analyze sentiment of messages
* Provide a simple prediction system

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* PyTorch
* Hugging Face Transformers

---

## 📂 Project Structure

```
spam_project/
│── data/                # Dataset
│── model/               # Trained model (ignored using .gitignore)
│── results/             # Output results
│── train.py             # Training script
│── predict.py           # Prediction script
│── app.py               # Application file
│── requirements.txt     # Dependencies
│── README.md            # Documentation
```

---

## 📊 Dataset

* SMS Spam Collection Dataset
* Columns:

  * `label` → spam / ham
  * `text` → message content
* Labels converted to:

  * Spam = 1
  * Ham = 0

---

## ⚙️ Installation

### 1. Clone Repository

```
git clone https://github.com/YOUR-USERNAME/spam-detection-project.git
cd spam_project
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Train Model

```
python train.py
```

### ▶️ Predict Messages

```
python predict.py
```

### ▶️ Run Application

```
python app.py
```

---

## 🤖 Model Details

* Model: XLM-RoBERTa
* Task: Text Classification
* Library: Hugging Face Transformers
* Framework: PyTorch

---

## 📈 Features

* Spam classification using deep learning
* Sentiment analysis integration
* Transformer-based NLP model
* Easy-to-run scripts

---

## ⚠️ Important Notes

* `model/` folder is ignored in Git
* Training on CPU may be slow
* GPU recommended for faster training

---

## 🔮 Future Enhancements

* Web app deployment (Flask/Streamlit)
* Larger dataset for better accuracy
* Multilingual spam detection
* Model optimization

---

## 📚 Conclusion

This project demonstrates how transformer-based models like XLM-RoBERTa can effectively detect spam and analyze sentiment in text data.

---

## 👨‍💻 Author

k.Sriram
