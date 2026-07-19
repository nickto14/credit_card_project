import pandas as pd 
import datetime as date

###############################################################
# Load in Data 
###############################################################

df = pd.read_csv('D:/Users/nickt/Credit_Card_Project/Data/CreditCardData.csv')

###############################################################
# We first change the type and reformat certian variables
################################################################ 

# Drop unnessesacry variable
df = df['Day of Week'].drop()

# Drop first character from Trade ID and Amount: 

df['Transaction ID'] = df['Transaction ID'].str[1:]
df['Amount'] = df['Amount'].str[1:]

# Change Data Type:

df['Transaction ID'] =  df['Transaction ID'].astype(int)
df['Amount'] =  df['Amount'].astype(int)
df['Date'] =  df['Date'].astype(date)




