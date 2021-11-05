import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """load the csv formatted messags and categories files, join on common id, and return as a dataframe
	
    Parameters:
    messages_filepath (str): path and name of the messages file containing data in a csv format
    categories_filepath (str): path and name of the categories file containing data in a csv format
      
    Returns:
    dataframe: dataframe object made by merging messages and categories
    
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    # Merge the messages and categories datasets using the common id
	
    df = messages.merge(categories, how='left', on='id')
    return df

def clean_data(df):
    """Clean the data by extracting features, and naming the features
    

    
    Use the first row of categories dataframe to create column names for the categories data.
    Rename columns of categories with new column names.

	
    Parameters:
    df (dataframe): dataframe formed by merging categories and messages on a common id
      
    Returns:
    dataframe object: cleaned dataframe consisting of feature set (categories columns)
    
    
    """
    # Split the values in the categories column on the ; character so that each value becomes a separate column. You'll find this
	# method very helpful! Make sure to set expand=True.
	# https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames


    for column in categories:
        # set each value to be the last character of the string
        val = categories[column].str[-1:]

		# specifically for the related column there are number of rows where the count is 2
		# this is to conver a value greater than 1 to 1, and applies to all columns
        if (val > 1) : 
        	val = 1
        
        categories[column] = val  # categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(keep='last', inplace=True)

    return df


def save_data(df, database_filename):
    """convert the dataframe into a sqllite DB and save as a filesystem DB file

	
    Parameters:
    df (dataframe): name of the dataframe
    database_filename (str): path and name of the file to save on filesystem
      
    Returns:
    dataframe object: cleaned dataframe consisting of feature set (categories columns)
    
    
    """
    engine = create_engine(os.path.join('sqlite:///', database_filename))
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)



def main():
    """module to read in the categories and messages csv formatted file, join them, clean them and save to a filesystem

	
    Parameters:
    messages file (str): name of the file, along with the path, containing the messages in a csv format
    categories file (str): name of the file, along with the path, containing the categories in a csv format
    db_name (str): name of the DB file, laong with the path, where to save the cleaned data as sqllite DB
      
    Returns:
	N/A
    
    
    """

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
