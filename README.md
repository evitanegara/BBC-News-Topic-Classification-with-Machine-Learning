![image](https://github.com/user-attachments/assets/bcb48583-95b4-42b1-8be0-1a5d42af0cdd)# BBC News Topic Classification Using Machine Learning

## Project Overview
This project focuses on the application of Natural Language Processing (NLP) and supervised machine learning to automate topic classification of BBC news articles. In the context of digital journalism and growing information overload, automated topic labelling helps streamline news delivery systems, improve content recommendations, and facilitate large-scale media analysis. Using a labeled dataset of over 2,200 articles from BBC News, this study evaluates three core classifiers: Naïve Bayes, Random Forest, and Support Vector Machine (SVM), tested with two feature extraction techniques: TF-IDF and Bag of Words (BoW). Each model is assessed through confusion matrices and classification reports (precision, recall, f1-score, accuracy), with performance visualized to compare classification strength across topics such as business, entertainment, politics, sport, and tech.

## Dataset Overview

- Source: [Kaggle - BBC Full Text Document Classification](https://www.kaggle.com/datasets)
- Total Records: 2,225 news articles
- Categories: Business, Entertainment, Politics, Sport, Tech
- The dataset is well-balanced, supporting fair model evaluation across all classes without the need for oversampling or downsampling.

| Category      | Document Count |
|---------------|----------------|
| Sport         | 511            |
| Business      | 510            |
| Politics      | 417            |
| Tech          | 401            |
| Entertainment | 386            |

## Executive Summary
This project demonstrates how NLP preprocessing and machine learning modeling can be used to automate the classification of news articles. The models were trained on 70% of the dataset and evaluated on the remaining 30%, using two different vectorization techniques. SVM with TF-IDF vectorization achieved the highest accuracy at 97.75%. Naïve Bayes using Bag-of-Words also reached 97.75%, showing strong performance even with simpler feature inputs. Random Forest with TF-IDF followed closely at 97.46%, offering a balance between accuracy and interpretability. Across all models, articles in the sport and business categories were identified with the highest precision. Categories such as entertainment and tech presented minor classification challenges due to overlapping vocabulary and contextual similarities.

# Project Overview : 
This project aims to classify BBC News articles into five categories using machine learning. Key steps include:
- **Data Import and Library Setup**:
  - Utilized libraries such as NumPy, scikit-learn, pandas, and NLTK for text processing.
  - Downloaded NLTK resources like Stopwords and WordNet to aid in text preprocessing.
- **Check Distribution of Data**: Examined the distribution of the five news categories: "sport" (511 documents), "business" (510 documents), "politics" (417 documents), "tech" (401 documents), and "entertainment" (386 documents).
- **Pre-Processing** : 
  - Lowercasing: Converted all text to lowercase to ensure consistency and reduce vocabulary size, improving the effectiveness of topic labelling.
  - Lemmatization: Applied WordNet Lemmatizer to convert words into their base form, removing inflectional endings.
  - Stop-word Removal: Eliminated common stop-words (e.g., articles, pronouns) to focus on meaningful words and reduce the dimensionality of the data.
    ![image](https://github.com/user-attachments/assets/8f0e8c95-3b6c-4c8d-b7fb-850f9531c6e6)

- **Train-Test Split**: Split the dataset into 70% training and 30% testing using the train_test_split function.
- **Feature Extraction**: Used TF-IDF to assess word relevance and Bag of Words to count word occurrences, both techniques aiding in topic classification.
- **Modeling and Predictions**:
  - Applied Naïve Bayes, Random Forest, and SVM for topic labelling.
  - Random Forest was chosen for its adaptability, SVM for its effectiveness in high-dimensional data, and Naïve Bayes for its simplicity and efficiency.
- **Evaluation**: Assessed the performance of the models using confusion matrices, accuracy, precision, recall, and F1-score metrics.
- **Project result** 
  - Random Forest: Achieved 97.46% accuracy with TF-IDF, and 96.56% with Bag of Words.
  - SVM: Reached 97.75% accuracy using TF-IDF, and 96.26% with Bag of Words.
  - Naïve Bayes: BoW model achieved 97.75% accuracy, while TF-IDF reached 96.71%.

## ⚙Project Workflow

### Data Import and Library Setup
- Imported core libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, and NLTK
- Downloaded stopwords and WordNetLemmatizer from NLTK for text preprocessing

### Text Preprocessing
- Lowercasing: Standardized text format to reduce vocabulary size
- Tokenization and Lemmatization: Split text into tokens and converted each to its root form using WordNet
- Stopword Removal: Removed common, non-informative words such as "the", "is", etc.
- This preprocessing pipeline reduced dimensionality and emphasized meaningful terms for vectorization
- 
### Train-Test Split
- Used `train_test_split` to divide the dataset into 70% training and 30% testing subsets
  
### Feature Extraction
- Applied two vectorization techniques: TF-IDF: Weighed words by term frequency and inverse document frequency to highlight discriminative features
- Bag-of-Words (BoW): Counted raw word occurrences, providing a simpler input format

### Model Building and Training
- Trained the following models using both TF-IDF and BoW features:
  - Naïve Bayes: Effective for frequency-based text classification
  - Random Forest: Ensemble of decision trees offering interpretability and balanced performance
  - Support Vector Machine (SVM): Suitable for high-dimensional data, maximizing class separation margins

### Model Evaluation
- Evaluation metrics included: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Plotted confusion matrices for both BoW and TF-IDF to visually compare classification performance across categories
![image](https://github.com/user-attachments/assets/4eda7247-c24a-423e-bffa-689605d6f153)
![image](https://github.com/user-attachments/assets/f7ff7356-56c1-4c3a-a2b2-22924edb3b6a)
![image](https://github.com/user-attachments/assets/88edf0aa-d8f2-4243-9984-e22cdd3d1cb6)
![image](https://github.com/user-attachments/assets/72883496-93cd-49ad-bec9-4368bd8d7ed6)
![image](https://github.com/user-attachments/assets/51996b39-c13d-4953-9eb2-2ef1342b81e0)
![image](https://github.com/user-attachments/assets/8b5e4168-db2a-428c-a802-3874baa377e5)


## 🔍 Highlights

- SVM with TF-IDF delivered top accuracy and precision across all categories
- Naïve Bayes with Bag-of-Words achieved matching accuracy using a simpler feature representation
- Random Forest with TF-IDF maintained strong consistency and interpretability
- All models performed best on the sport category, and weakest on tech due to semantic overlap in vocabulary

---

## Key Takeaways

- Preprocessing matters  
  Token normalization and stopword removal significantly improved feature quality

- TF-IDF excels in precision  
  Especially effective for distinguishing categories with similar vocabulary patterns

- Model synergy  
  Each algorithm performed best when paired with suitable feature types (SVM with TF-IDF, Naïve Bayes with BoW)

- No resampling needed  
  Balanced class distribution enabled fair and unbiased model evaluation

---

# Contact
For any questions or inquiries, please contact evitanegara@gmail.com
