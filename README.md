# Business Understanding 
As the internet grows, it has become a vast repository of information, particularly news articles, which play a significant role in daily life. However, organizing the enormous volume of news manually is inefficient and time-consuming. Therefore, automating the categorization of news into specific topics is crucial for better management and retrieval.

# Problem Objective : 
- The goal of this project is to classify BBC News articles into five categories using machine learning-based topic labelling.
- Machine learning models applied include Naïve Bayes, Random Forest, and Support Vector Machine (SVM).
- By automating news categorization, this project enhances the efficiency of information organization, making it easier to search, retrieve, and analyze large volumes of text data.
- The dataset used can be accessed [here](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data).


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
  
# Conclusion 
All classifiers performed exceptionally well, with Random Forest and SVM showing enhanced accuracy with TF-IDF, while Naïve Bayes excelled with Bag of Words. Feature extraction techniques played a significant role, as TF-IDF improved predictive capacity by emphasizing unique terms, whereas Bag of Words simplified the process. In summary, the combination of these models and feature extraction methods led to robust and highly accurate classification models, achieving accuracies over 96%.
