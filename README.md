#  BBC News Topic Classification with Machine Learning
## Project Overview
This project focuses on the application of Natural Language Processing (NLP) and supervised machine learning to automate topic classification of BBC news articles. In the context of digital journalism and growing information overload, automated topic labelling helps streamline news delivery systems, improve content recommendations, and facilitate large-scale media analysis. Using a labeled dataset of over 2,200 articles from BBC News, this study evaluates three core classifiers: Naïve Bayes, Random Forest, and Support Vector Machine (SVM), tested with two feature extraction techniques: TF-IDF and Bag of Words (BoW). Each model is assessed through confusion matrices and classification reports (precision, recall, f1-score, accuracy), with performance visualized to compare classification strength across topics such as business, entertainment, politics, sport, and tech.

## Dataset Overview

- Source: Kaggle - BBC Full Text Document Classification
- Total Records: 2,225 news articles
- Categories: Business, Entertainment, Politics, Sport, Tech

| Category      | Document Count |
|---------------|----------------|
| Sport         | 511            |
| Business      | 510            |
| Politics      | 417            |
| Tech          | 401            |
| Entertainment | 386            |

## Executive Summary
This project demonstrates how NLP preprocessing and machine learning modeling can be used to automate the classification of news articles. The models were trained on 70% of the dataset and evaluated on the remaining 30%, using two different vectorization techniques. SVM with TF-IDF vectorization achieved the highest accuracy at 97.75%. Naïve Bayes using Bag-of-Words also reached 97.75%, showing strong performance even with simpler feature inputs. Random Forest with TF-IDF followed closely at 97.46%, offering a balance between accuracy and interpretability. Across all models, articles in the sport and business categories were identified with the highest precision. Categories such as entertainment and tech presented minor classification challenges due to overlapping vocabulary and contextual similarities.

## Project Workflow

### Data Import and Library Setup
- Imported core libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, and NLTK
- Downloaded stopwords and WordNetLemmatizer from NLTK for text preprocessing

### Text Preprocessing
- Lowercasing: Standardized text format to reduce vocabulary size
- Tokenization and Lemmatization: Split text into tokens and converted each to its root form using WordNet
- Stopword Removal: Removed common, non-informative words such as "the", "is", etc.
- This preprocessing pipeline reduced dimensionality and emphasized meaningful terms for vectorization
 <p align="center">
  <img src="https://github.com/user-attachments/assets/b05289bc-f502-4e0f-a455-bda57f7f2555" alt="Forecasting" width="500"/>
</p>

### Train-Test Split
- Used train_test_split to divide the dataset into 70% training and 30% testing subsets
  
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

 <p align="center">
  <img src="https://github.com/user-attachments/assets/e002ecab-8866-47ce-9d7b-c6f9a21818bf" alt="Forecasting" width="500"/>
</p>
 <p align="center">
  <img src="https://github.com/user-attachments/assets/2f52f776-a10a-4767-b59b-cd2370b88c88" alt="Forecasting" width="500"/>
</p>
 <p align="center">
  <img src="https://github.com/user-attachments/assets/a1bb5594-dabb-4f4f-8928-2fbe9b5a8f02" alt="Forecasting" width="500"/>
</p>

## Highlights
- SVM with TF-IDF delivered the highest accuracy (97.75%), excelling in precision and recall across all five categories—especially effective in distinguishing semantically close classes like business and tech.
- Naïve Bayes with Bag-of-Words achieved a matching accuracy (97.75%) despite its simplicity, showcasing strong generalization and high recall on the politics and entertainment categories.
- Random Forest with TF-IDF maintained robust performance (97.46%) with a strong balance of interpretability and accuracy, particularly excelling in sport and business classifications.
- Sport category was most consistently predicted correctly across all models, indicating strong signal clarity and distinguishable vocabulary patterns.
- Tech category posed the most challenges, often confused with business due to shared terminology and overlapping content themes.

## Key Takeaways
- Model-Feature Synergy Matters: SVM paired best with TF-IDF, while Naïve Bayes performed best with BoW—highlighting the importance of aligning model type with feature representation.
- TF-IDF Improves Precision: Particularly effective in filtering out common terms and amplifying discriminative features for nuanced topics.
- No Resampling Required: Balanced class distribution allowed for fair comparison without needing under/over-sampling methods.
- Preprocessing Drives Performance: Lemmatization, lowercasing, and stopword removal significantly reduced noise and improved classification clarity.

# Contact
For any questions or inquiries, please contact evitanegara@gmail.com
