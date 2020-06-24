from faker import Faker
from sklearn.utils import shuffle
import pandas as pd


DataGenerator = Faker()
DataGenerator.random.seed(5467)

def generate_dataset(num_records, ratio, path):
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
        , 'rules_engine_prediction'
        , 'fraud_label'))
    num_non_fraud = int(round(num_records * (1 - ratio)))
    num_fraud = int(round(num_records * ratio))

    for i in range(num_non_fraud):
        if i % 100 == 0: print("{} / {}".format(i, num_non_fraud))
        row = [DataGenerator.random_int(min=100000, max=999999)
            , DataGenerator.random_int(min=9000, max=9200)
            , DataGenerator.random_int(min=1000, max=1200)
            , DataGenerator.zipcode()
            , DataGenerator.random_int(min=1000, max=9999)
            , DataGenerator.company()
            , DataGenerator.random.choice(['4021', '2333', '2002', '4050', '7383', '9832', '9883'])
            , DataGenerator.zipcode()
            , DataGenerator.bank_country()
            , DataGenerator.random_int(min=1, max=2500)
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(['M', 'P', 'N'])
            , 0
            , 0]
        df.loc[i] = [item for item in row]

    # fraud
    for i in range(num_fraud):
        if i % 100 == 0: print("{} / {}".format(i, num_fraud))
        row = [DataGenerator.random_int(min=100000, max=999999)
            , DataGenerator.random_int(min=9000, max=9200)
            , DataGenerator.random_int(min=1000, max=1200)
            , DataGenerator.zipcode()
            , DataGenerator.random_int(min=1000, max=9999)
            , DataGenerator.company()
            , '1011'
            , DataGenerator.zipcode()
            , DataGenerator.bank_country()
            , DataGenerator.random_int(min=1, max=2500)
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(["A", "B", "C", "D"])
            , DataGenerator.random.choice(['M', 'P', 'N'])
            , DataGenerator.random.choice([0,1])
            , 1]
        df.loc[num_non_fraud + i] = [item for item in row]

    df = shuffle(df)
    df.to_csv(path, sep=',', encoding='utf-8', header=True, index=False, mode='a')
    return df


df = generate_dataset(1000, 0.1, 'transactions_with_rules_engine.csv')