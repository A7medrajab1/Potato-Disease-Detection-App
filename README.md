# 🥔 Potato Disease Detection

A deep learning web application for detecting diseases in potato leaves using image classification.  
Supports **Early Blight**, **Late Blight**, and **Healthy** classes.

## 🚀 Features
- 🔍 Detect three potato leaf conditions
- 🤖 Two prediction models:
  - **InceptionV3** (high accuracy)
  - **Custom CNN**
- 📊 Confidence score visualization
- ✂️ Optional image cropping
- ℹ️ Disease symptoms + treatment tips
- 🎨 Simple and clean Streamlit interface

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/A7medrajab1/Potato-Disease-Detection-App.git
cd potato-disease-detection
2. Create & activate a virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate    # Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Add trained models
Place your models inside:

bash
Copy code
Models/inception_savedmodel/
Models/cnn_savedmodel/
▶️ Usage
Run the app:

bash
Copy code
streamlit run app.py
The interface will open at:

arduino
Copy code
http://localhost:8501
📁 Project Structure
├── app.py
├── requirements.txt 
├── README.md 
│
├── Models/
│   ├── inception_savedmodel/ # InceptionV3 model
│   └── cnn_savedmodel/       # Custom CNN model
│
├── Data/
│   ├── confusion_Inception.png
│   └── confusion_CNN.png
│
├── notebooks/
│
├── Img_for_test/ 
│
└── demo.mp4


📦 Dataset
Trained on the PlantVillage dataset : https://www.kaggle.com/datasets/arjuntejaswi/plant-village

Early Blight

Late Blight

Healthy

📜 License
MIT License.

👤 Contact
GitHub: A7medrajab1

Email: ahmedelzaiaty2004@gmail.com