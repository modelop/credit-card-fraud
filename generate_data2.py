from faker import Faker
from sklearn.utils import shuffle
import pandas as pd


DataGenerator = Faker()
DataGenerator.random.seed(5467)

def generate_dataset(path):
    df = pd.DataFrame(columns=(
        'transaction_id'
        , 'card_id'
        , 'customer_id'
        , 'customer_zipcode'
        , 'merchant_id'
        , 'merchant_name'
        , 'merchant_category'
        , 'merchant_zipcode'
        , 'merchant_country'
        , 'transaction_amount'
        , 'authorization_response_code'
        , 'atm_network_xid'
        , 'cvv_2_response_xflg'
        , 'fraud_label'))

#fraud
    for i in range(10):

        row = [DataGenerator.random_int(min=100000, max=999999)
            , DataGenerator.random_int(min=9000, max=9200)
            , DataGenerator.random_int(min=1000, max=1200)
            , DataGenerator.zipcode()
            , DataGenerator.random_int(min=1000, max=9999)
            , DataGenerator.company()
            , '7777'
            , DataGenerator.zipcode()
            , DataGenerator.bank_country()
            , DataGenerator.random_int(min=1, max=2500)
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(['M', 'P', 'N'])
            , 1]
        df.loc[i] = [item for item in row]

    df = shuffle(df)
    df.to_csv(path, sep=',', encoding='utf-8', header=True, index = False,  mode='a')
    return df


df = generate_dataset('new_fraud_transactions.csv')