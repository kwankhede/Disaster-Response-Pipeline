import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    A function to load a dataset
    
    Input:  
         messages_filepath   : Filepath of a first file
         categories_filepath : Filepath of a second file
    Output: 
      Dataframe 
   
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge two datasets
    df = pd.merge(messages, categories)
    return df 
    
   

def clean_data(df):
    
    '''
    A function to clean textdata
    
    Input : 
           df = dataframe 
    Output :
         
          df = cleaned data
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x[:-2] , row))
    
    # Rename the columns  
    categories.columns = category_colnames
    
    for column in categories:
      # set each value to be the last character of the string
        categories[column] = list(map(lambda x: x[-1:] ,  categories[column]))
    
      # convert column from string to int
        categories[column] =  categories[column].astype(int)
        
    # drop categories column from the dataframe
    df = df.drop('categories', axis = 1)
    
    # Join two dataframes. 
    df = pd.concat([df, categories], axis = 1)
    
    #Drop duplicate entries
    df = df.drop_duplicates()
    return df 


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('Disaster_messages', engine) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()