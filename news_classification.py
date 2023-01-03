# Import libraries
import pandas as pd
import logging as lg
from data_preprocessing import preprocess_text
from data_splitting import data_splitting
from data_encoder import onehotencoder,labelencoder
from tokenizer_padding import tokenizer_and_padding

# Logging
lg.basicConfig(filename='All_logs/logs.log',
               filemode='a',
               format='%(asctime)s-%(levelname)s-%(message)s',
               datefmt='%Y-%m-%d %H:%M:%S',
               level=lg.INFO
               )

lg.info('DATA LOADING')
# Import dataset
try:
    # Load train dataset
    lg.info('Load Train dataset')
    df = pd.read_csv('Dataset/BBC News Train.csv')
except Exception as e:
    raise e



# Factorize the category label with numeric
lg.info('Factorize the category label with numeric')
df['category_id'] = df.Category.factorize()[0]


# Check Head
lg.info(f'Check first five rows: \n{df.head()}')
print(df.head())
print()
# Check tail
lg.info(f'Check last five rows: \n{df.tail()}')
print(df.tail())

# Check shape
lg.info(f'Dataset shape: {df.shape}')
print(f'Dataset shape: {df.shape}')

lg.info('*'*100)

lg.info('DATA CLEANING')


# Apply data_preprocessing funtion to our train data
lg.info('Apply preprocessing function to clean dataset')
df.Text = df.Text.apply(preprocess_text)

# Save 'cleaned' dataset
lg.info('Saved cleaned dataset')
df.to_csv('cleaned.csv')

lg.info('*'*100)

# Splitting
lg.info('DATA SPLITTING')
lg.info('Load cleaned.csv dataset')
df_clean = pd.read_csv('cleaned.csv')

# Independent features
lg.info('Get Independent features')
X = df_clean['Text'].values

# Dependent features
lg.info('Get Dependent features')
y = df_clean['Category'].values

# Data splitting
lg.info('Split the "Dependent" and "Independent" features as train and test sets')
X_train_sent, X_test_sent, y_train, y_test = data_splitting(X=X, y=y)

# Check the distribution of train and test set
lg.info(f"The length of train set features is : {len(X_train_sent)}")
print(f"The length of train set features is : {len(X_train_sent)}")
lg.info(f"The length of test set features is : {len(X_test_sent)}")
print(f"The length of test set features is : {len(X_test_sent)}")
lg.info(f"The length of train set labels is : {len(y_train)}")
print(f"The length of train set labels is : {len(y_train)}")
lg.info(f"The length of test set labels is : {len(y_test)}")
print(f"The length of test set labels is : {len(y_test)}")

lg.info('*'*100)

lg.info('CATEGORICAL ENCODING')
lg.info('One Hot Encoder for labels')
y_train_one_hot, y_test_one_hot = onehotencoder(y_train, y_test)
y_train_label, y_test_label, class_names = labelencoder(y_train, y_test)

lg.info('Print the first 10 samples of the encoded labels')
lg.info(f'y_train_one_hot: \n{y_train_one_hot[:10]}')
print(f'y_train_one_hot: \n{y_train_one_hot[:10]}')
lg.info(f"y_test_one_hot: \n{y_test_one_hot[:10]}")
print(f"y_test_one_hot: \n{y_test_one_hot[:10]}")
lg.info(f"y_train_label:{y_train_label[:10]}")
print(f"y_train_label:{y_train_label[:10]}")
lg.info(f'y_test_label:{y_test_label[:10]}')
print(f'y_test_label:{y_test_label[:10]}')
lg.info(f'Class Names:{class_names}')
print(f'Class Names: {class_names}')
lg.info('*'*100)

lg.info('TOKENIZER AND PADDING')
# Get X_train and X_test sentences mapped to tokenizer and padding
lg.info('Get X_train and X_test sentences mapped to tokenizer and padding')
X_train, X_test, _ = tokenizer_and_padding(X_train_sent, X_test_sent)

lg.info(f'Print first 5 samples from X_train and X_test ')
lg.info(f'X_train: {X_train[:5]}')
print(f'X_train: {X_train[:5]}')
lg.info(f'X_test: {X_test[:5]}')
print(f'X_test: {X_test[:5]}')
print()
lg.info('X_train and X_test shape')
lg.info(f'X_train shape: {X_train.shape}')
print(f'X_train shape:{X_train.shape}')
lg.info(f'X_test shape: {X_test.shape}')
print(f'X_test shape:{X_test.shape}')