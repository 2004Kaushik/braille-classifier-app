# 🔠 Braille Character Classifier API

An API that classifies Braille characters from `.jpg` images using a deep learning model built with TensorFlow/Keras. Deployed using FastAPI inside a Docker container for portability and ease of cloud deployment.

> 🚧 **Work in Progress**: This project is currently under development. Features, model performance, and API endpoints may change.

---

## 📦 Features

- Upload a `.jpg` image of a Braille character.
- Preprocesses and feeds the image into a trained CNN model (`braille_classifier.keras`).
- Returns the predicted class along with a confidence score.
- REST API built with **FastAPI**.
- Dockerized for simple local or cloud deployment.

---

## 🧠 Model

The model (`braille_classifier.keras`) is trained to classify Braille characters based on their image representations. It expects RGB images resized to a defined input size (e.g., `64x64`).

---

## 📁 Project Structure

braille-api/ 
│
├── app.py # FastAPI server with /predict endpoint
├── braille_classifier.keras # Trained TensorFlow model
├── requirements.txt # Dependency list
├── Dockerfile # Docker container setup
└── README.md # Project documentation
