# Deep Learning Nanodegree - Udacity (PyTorch)

This repository contains materials related to Udacity's [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It includes project implementations, learning resources, and supporting materials covering deep learning topics such as **Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, and Generative Adversarial Networks (GANs)**.

Each project in this repository reflects the **hands-on work** done to complete the Nanodegree, demonstrating practical applications of deep learning in **computer vision, natural language processing (NLP), and generative modeling**.

---

## 📌 **Projects Overview**

### **2️⃣ Intro to Deep Learning with PyTorch**  
**Project: Developing a Handwritten Digits Classifier with PyTorch**  
- Implemented a **fully connected neural network** to classify handwritten digits.
- Trained on the **MNIST dataset** and optimized using PyTorch.
- Developed a **proof of concept for Optical Character Recognition (OCR)** using deep learning.
- Preprocessed and augmented data to improve recognition accuracy.
- Tuned hyperparameters for optimal model performance.

### **3️⃣ Convolutional Neural Networks (CNNs)**  
**Project: Landmark Classification and Tagging for Social Media**  
- Built a **CNN model** to classify **landmarks** in images.
- Applied **data augmentation** and **transfer learning** techniques.
- Developed a model to infer image locations based on landmarks when metadata is unavailable.
- Compared the accuracy of different CNN architectures and selected the best-performing model.
- Deployed the trained model to simulate real-world applications, such as automatic photo tagging.

### **4️⃣ Recurrent Neural Networks (RNNs) & Transformers**  
**Project: Text Translation and Sentiment Analysis using Transformers**  
- Developed an **NLP model** to analyze **sentiment of movie reviews** in three languages: **English, French, and Spanish**.
- Merged multiple CSV files containing movie reviews, synopses, and metadata into a unified **pandas dataframe**.
- Used **HuggingFace Transformers** to translate French and Spanish reviews into English for unified analysis.
- Applied **pretrained transformers** to perform **sentiment classification (Positive/Negative)** on each review.
- Generated a final **CSV report** containing columns for **Title, Year, Synopsis, Review, Sentiment, and Original Language**.

### **5️⃣ Generative Adversarial Networks (GANs) - Face Generation**  
**Project: Face Generation - Training a GAN on CelebA**  
- Created a **DCGAN (Deep Convolutional GAN)** to generate realistic human faces using the CelebA dataset.
- Built a **data pipeline** to preprocess CelebA images, including cropping and resizing to 64x64x3.
- Designed and implemented a **custom discriminator** to classify real vs. fake images.
- Built a **generator model** that takes a latent vector as input and produces high-resolution 64x64x3 images.
- Applied **ConvTranspose2d layers** for upscaling latent vectors and **Batch Normalization** for stable training.

---

## 📚 **Topics Covered**
✔️ **Neural Networks** - Activation functions, optimization techniques, and backpropagation.  
✔️ **CNNs** - Feature extraction, pooling layers, and pre-trained models.  
✔️ **RNNs & Transformers** - Sequence modeling, LSTMs, GRUs, and self-attention.  
✔️ **GANs** - Adversarial training, DCGAN, and image generation.  
✔️ **Transfer Learning** - Leveraging pre-trained models for new tasks.  
✔️ **NLP Processing** - Sentiment analysis, text translation, and data augmentation.  
✔️ **OCR** - Optical Character Recognition using deep learning.  

---

## 🚀 **Setup & Installation**
To run the projects locally, follow these steps:

1. **Clone the repository**
   ```sh
   git clone https://github.com/Joe-ElM/Deep-Learning-ND.git
   cd Deep-Learning-ND
   ```

2. **Create a virtual environment**
   ```sh
   conda create -n deep-learning python=3.8
   conda activate deep-learning
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**
   ```sh
   jupyter notebook
   ```

---

## 🎯 **Next Steps**
✅ Expand projects with additional **fine-tuning & hyperparameter optimization**.  
✅ Implement **real-world applications** using these models.  
✅ Explore **deployment strategies** for production-level deep learning applications.  

---

### 🔗 **Resources**
- [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [TensorFlow & Keras](https://www.tensorflow.org/)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

This repository serves as a **comprehensive reference** for my work in deep learning. Contributions and feedback are welcome! 🚀
