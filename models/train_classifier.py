import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import roc_auc_score , precision_score, recall_score, f1_score, classification_report,accuracy_score,confusion_matrix,roc_curve, auc


def load_data(database_filepath):
    
    '''
    A function to load a table from SQL_database
    
    Input: database_filepath
    Output: X, Y and message categories
    
    '''
    # load data from database
    table_name = 'Disaster_messages'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name,engine)
    
    # Separate df into X and Y
    X = df.message
    Y = df[df.columns[5:]]
    dummies = pd.get_dummies(df[['related','genre']])
    Y = pd.concat([Y, dummies], axis=1)
    category_names = Y.columns
    
    return X,Y, category_names
    


def tokenize(text):
    '''
    A fucntion to tokenizing a given text.
    
    Input  : text
    Output : A list of clean tokens 
    
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #Detct all urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Split the text into tokens
    tokens = word_tokenize(text)
    
    # Chnage words into their basic form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    model = Pipeline([
       ('vect', CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('model', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced")))
       ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    A function to print classification report
    
    Input:  
          Y_test 
          Y_pred 
    
    Output: 
          Classification
    '''
    # Predict
    Y_pred = model.predict(X_test)
    
    results = pd.DataFrame([[0, 0,0,0]],columns=['Message_type', 'f_score', 'precision', 'recall'])
    j = 0
    for i in range(len(category_names)):
        prec = precision_score(Y_test.iloc[:, i].values, Y_pred[:, i],average='weighted')
        rec = recall_score(Y_test.iloc[:, i].values, Y_pred[:, i],average='weighted')
        f1  = f1_score(Y_test.iloc[:, i].values, Y_pred[:, i],average='weighted')
        model_results =  pd.DataFrame([[i, f1,prec,rec]],
                                      columns=['Message_type', 'f_score', 'precision', 'recall'])
        results = results.append(model_results, ignore_index = True)
    print('Weighted_average f_score:', results['f_score'].mean())
    print('Weighted_averageprecision:', results['precision'].mean())
    print('Weighted_average recall:', results['recall'].mean())
    return results[1:]


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()