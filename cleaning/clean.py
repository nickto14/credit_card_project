import pandas as pd 
import datetime as date

path = 'D:/Users/nickt/Credit_Card_Project/Data'

###############################################################
# Load in Data 
###############################################################

df = pd.read_csv(f'{path}/CreditCardData.csv')

###############################################################
# We first change the type and reformat certian variables
###############################################################

# Fix mispelled character 
df["Bank"] = df["Bank"].replace({"Barlcays": "Barclays"})
df["Time"] = df["Time"].replace({24: 0})

# Drop unnessesacry variable
df = df.drop(columns = ['Date', 'Day of Week', 'Transaction ID'])

# Drop first character from Amount:
df['Amount'] = df['Amount'].str[1:]

# Change Data Type:
df['Amount'] = df['Amount'].astype(float)

###############################################################
# Create binary "mismatch" features. These flag when related location 
# variables disagree which is a strong signal for detecting fraud.
###############################################################

# If country of transaction and country of residence mismatch
df['trans_country_mismatch'] = (df['Country of Transaction'] != df['Country of Residence']).astype(int)

# If country of shipping and country of residence mismatch
df['shipping_residence_mismatch'] = (df['Shipping Address'] != df['Country of Residence']).astype(int)

# If country of transaction and country of shipping
df['trans_shipping_mismatch'] = (df['Country of Transaction'] != df['Shipping Address']).astype(int)

###############################################################
# Because some of the columns are strings and we want numerical
# values, we use one hot encoder to turn words into numbers
# our models can understand.
###############################################################

categorical_cols = ['Type of Card', 'Entry Mode', 'Type of Transaction', 
                    'Merchant Group', 'Country of Transaction', 'Shipping Address',
                    'Country of Residence', 'Gender', 'Bank']

df_encoded = pd.get_dummies(df, columns = categorical_cols, drop_first = True, dtype = int)

###############################################################
# Export Data
###############################################################

df_encoded.to_csv(f'{path}/CreditCardData_cleaned.csv', index=False)
