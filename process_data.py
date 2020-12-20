#
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



def load_data(general_df_path, customers_df_path):

    #load general dataset
    azdias = pd.read_csv(general_df_path, sep=';', dtype={'CAMEO_DEUG_2015': str, 'CAMEO_INTL_2015':str})
    customers = pd.read_csv(customers_df_path, sep=';', dtype={'CAMEO_DEUG_2015': str, 'CAMEO_INTL_2015':str})

    azdias['mailout'] = False
    customers['mailout'] = True

    # Store LNR values
    others_id = azdias['LNR'].unique()
    customers_id = customers['LNR'].unique()

    # Concatenate both datasets
    total_df = pd.concat([azdias, customers], axis=0, join='inner')

    return total_df, others_id, customers_id

def drop_na_columns(df, na_perct=.9):

    '''Drop columns with missing values when they represent 90 percent of the rows and return it.

    inputs
    na_perct(float): the percentage of missing values in a column.
    df(pandas DataFrame): the DataFrame

    output
    new_df(pandas DataFrame): the DataFrame after columns dropped.


    '''
    drop_cols = df.isnull().sum() / df.shape[0]
    labels = drop_cols[drop_cols>=na_perct].index.values
    new_df = df.drop(labels=labels, axis=1 )

    return new_df


def drop_na_rows(df, na_perct=.9):

    '''Drop rows with missing values when they represent 90 percent of the rows and return it.

    inputs
    na_perct(float): the percentage of missing values in a row.
    df(pandas DataFrame): the DataFrame

    output
    new_df(pandas DataFrame): the DataFrame after rows dropped.


    '''

    thresh = np.ceil(na_perct * df.shape[1])
    new_df = df.dropna(axis=0, thresh=thresh)

    return new_df

def drop_invariability(df, na_perct=0.9):
    '''
    Drop rows with more than na_perct pf invariability and return the new dataframe.
    '''
    cols = df.columns
    no_var_cols = []
    new_df = df

    for col in cols:
        aux = df[col].value_counts() / df[col].value_counts().sum()
        if np.sum(aux >= na_perct) > 0:
            new_df = new_df.drop(labels=col, axis=1)

    return new_df


def clean_data(df, drop_obj=True):

    # Replace non-numeric values
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].replace('\D', '-1', regex=True)
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].replace('\D+', '-1', regex=True)

    if drop_obj == True:
        # Drop columns with object type
        aux = (df.dtypes=='object')
        obj_cols = aux[aux].index.values

        df = df.drop(columns=obj_cols)

    new_df = drop_na_columns(df=df, na_perct=0.7)
    new_df = drop_na_rows(df=new_df, na_perct=0.7)
    new_df = drop_invariability(new_df, na_perct=0.9)

    new_df.reset_index(drop=True, inplace=True)

    return new_df

def impute_values(df, cols_with_missing):
    ''' Return a new dataframe with imputed values in numeric and categotical colums.

    inputs:
    df(pandas.DataFrame): The Dataframe
    strategy(str):

    output(pandas.DataFrame): imputed dataframe.
    '''
    # Dataframe with missing values
    mis_df = df[cols_with_missing]

    # Columns with numerical values
    num_df = mis_df.loc[:,mis_df.dtypes != 'object'].copy()
    num_cols = num_df.columns

    # Imputation in numerical column
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_num_df = pd.DataFrame(my_imputer.fit_transform(num_df))
    imputed_num_df.columns = num_cols

    # Columns with categoric values
    #cat_df = mis_df.loc[:,mis_df.dtypes == 'object'].copy()
    #cat_cols = cat_df.columns.values
    #df = df.drop(columns=cat_cols)

    # Join each dataframe
    total_cols = df.columns.values
    no_imputed_cols = np.setdiff1d(total_cols, num_cols)
    end_df = df[no_imputed_cols].join(imputed_num_df)

    return end_df


def save_data(df, df_filename):
    # Store the clean data
    df.to_csv(df_filename, header=df.columns, index=False)

def main():
    print("Loading data...\n  ")
    df, others_id, customers_id = load_data('data/Udacity_AZDIAS_052018.csv', 'data/Udacity_CUSTOMERS_052018.csv')
    print(df.head())

    print("\n Cleaning data ... \n")
    clean_df = clean_data(df, drop_obj=True)
    print(df.head())

    # Columns and dataframe with missing values
    aux = clean_df.isnull().sum().sort_values()
    cols_with_missing = aux[aux>0]
    cols_with_missing = cols_with_missing.index.values

    for col in cols_with_missing:
        if clean_df[col].isnull().sum() > 0.1*clean_df.shape[0]:
            clean_df[col+'_was_missing'] = clean_df[col].isnull()

    print("\n Imputing nan values ... \n")

    imputed_df = impute_values(df=clean_df, cols_with_missing=cols_with_missing)
    print(imputed_df.head())

    print('\n Saving data ... \n')
    save_data(imputed_df, 'total_df.csv')
    print("The dataset was stored!")

    #others_id = np.intersect1d(others_id, df['LNR'].unique())
    #customers_id = np.intersect1d(customers_id, df['LNR'].unique())

    #np.savetxt('others_id.out', others_id)
    #np.savetxt('customers_id.out', customers_id)


if __name__ == '__main__':
    main()
