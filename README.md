# Facial Emotion Recognition System 🎭

A deep learning computer vision project that identifies seven distinct human emotions from facial images using a custom Convolutional Neural Network (CNN).

## 📌 Project Overview
This project builds a robust classifier capable of detecting the following emotions from grayscale facial images:
* Angry 
* Disgust 
* Fear 
* Happy 
* Neutral 
* Sad 
* Surprise

The model addresses common challenges in facial recognition, specifically **class imbalance** and **overfitting**, by utilizing class weighting strategies and a streamlined architecture with Global Average Pooling.

## 📊 Dataset
* **Source:** FER-2013 (Facial Expression Recognition) Dataset.
* **Input Shape:** 48x48 pixels (Grayscale).
* **Structure:**
    * `train`: ~28,700 images.
    * `test`: ~7,100 images.
* **Data Augmentation:** To improve generalization, the training data is augmented in real-time using:
    * Random rotations (30 degrees).
    * Horizontal flipping.
    * Rescaling (1./255).

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Preprocessing:** Scikit-Learn (Class Weights)

## 🧠 Model Architecture
The model is a custom **Convolutional Neural Network (CNN)** designed from scratch:

1.  **Input Layer:** 48x48x1 Grayscale images.
2.  **Convolutional Blocks (x4):**
    * Features increasing filters (256 -> 128 -> 128 -> 64) to capture hierarchical features.
    * **Batch Normalization** applied after every convolution for stable training.
    * **MaxPooling2D** for dimensionality reduction.
3.  **Global Average Pooling:** Replaces the traditional `Flatten` layer to significantly reduce parameter count and minimize overfitting.
4.  **Dense Head:**
    * Fully connected layers (512 -> 64 units).
    * **Dropout (0.2)** for regularization.
5.  **Output Layer:** Softmax activation for 7-class classification.

## ⚖️ Handling Class Imbalance
The dataset is highly imbalanced (e.g., 'Happy' has many samples, while 'Disgust' has very few). To address this:
1.  **Class distribution** was analyzed and plotted.
2.  **Class Weights** were computed using `sklearn.utils.class_weight`.
3.  These weights were passed into the model training (`model.fit`) to penalize misclassification of minority classes more heavily.

## 📈 Results & Performance
* **Training Accuracy:** ~63%
* **Validation Accuracy:** ~60%
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy

### Evaluation Insights
* **Strengths:** The model performs excellently on **Happy** and **Surprise** emotions (high Precision/Recall).
* **Challenges:** Distinguishing between **Fear** and **Sadness** remains challenging due to overlapping facial features (a known benchmark issue with FER-2013).

## 🚀 How to Run
1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Run the Notebook:**
    Execute the cells in `Facial_Emotion_Detection.ipynb` to train the model.
4.  **Inference:**
    Use the `predict_emotion` function to test on new images:
    ```python
    prediction = predict_emotion('path/to/image.jpg')
    print(prediction)
    ```

## 🔮 Future Improvements
* **Transfer Learning:** Implement ResNet50 or MobileNet for potentially higher accuracy.
* **Facial Landmarks:** Integrate Dlib to crop faces more accurately before feeding them into the CNN.
* **Fine-tuning:** Unfreeze layers of a pre-trained model to capture more subtle facial micro-expressions.

---
*Created by [Ankit]*
