import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer




def load_data(general_df_path, customers_df_path):

    #load general dataset
    azdias = pd.read_csv(general_df_path, sep=';', dtype={'CAMEO_DEUG_2015': str, 'CAMEO_INTL_2015':str})
    customers = pd.read_csv(customers_df_path, sep=';', dtype={'CAMEO_DEUG_2015': str, 'CAMEO_INTL_2015':str})

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


def clean_data(df):

    # Replace non-numeric values
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].replace('\D', '-1', regex=True)
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].replace('\D+', '-1', regex=True)

    # Drop categoric columns with more than 2 categories
    df = df.drop(columns=['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM'])

    # Drop columns and rows with more than 70% of NaNs
    new_df = drop_na_columns(df=df, na_perct=0.7)
    new_df = drop_na_rows(df=new_df, na_perct=0.7)

    # Drop columns with more than 90% of invariability
    new_df = drop_invariability(new_df, na_perct=0.9)

    new_df.reset_index(drop=True, inplace=True)

    return new_df

def impute_values(df, cols_with_missing):
    ''' Return a new dataframe with imputed values in numeric and categotical colums.

    inputs:
    df(pandas.DataFrame): The Dataframe
    output:
    new_df(pandas.DataFrame): imputed dataframe.
    '''
    # Dataframe with missing values
    mis_df = df[cols_with_missing].copy()

    # Imputation in numerical column
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_df = pd.DataFrame(my_imputer.fit_transform(mis_df))
    imputed_df.columns = cols_with_missing

    # Join each dataframe
    total_cols = df.columns.values
    no_imputed_cols = np.setdiff1d(total_cols, cols_with_missing)
    new_df = df[no_imputed_cols].join(imputed_df)

    return new_df

def encode_nans(df, thresh=0.1):
    """Return a new dataframe with additional boolean columns indicating if there is missing values in a specific
    column acoording with the treshold of missing values, and return a 1d array with columns with missing values.

    inputs:
    df (pandas.Dataframe): the dataframe
    thersh (float): percentage of missing values in a column.

    oututs:
    new_df(pandas.Dataframe): new dataframe.
    cols_with_missing(numpy.ndarray): array with missing columns.

    """

    aux = df.isnull().sum().sort_values()
    cols_with_missing = aux[aux>0]
    cols_with_missing = cols_with_missing.index.values
    new_df = df.copy()

    for col in cols_with_missing:
        if df[col].isnull().sum() > thresh*df.shape[0]:
            new_df[col+'_was_missing'] = df[col].isnull()

    return new_df, cols_with_missing


def save_data(df, df_filename):
    # Store the clean data
    df.to_csv(df_filename, header=df.columns, index=False)


def main():
    print("Loading data...\n  ")
    df, others_id, customers_id = load_data('data/Udacity_AZDIAS_052018.csv', 'data/Udacity_CUSTOMERS_052018.csv')
    print(df.head())

    print("\n Cleaning data ... \n")
    clean_df = clean_data(df)
    print(clean_df.head())

    # Encode NaNs
    new_df, cols_with_missing = encode_nans(clean_df, thresh=0.1)
    print(new_df.head())
    # Imputing NaN values

    print("\n Imputing nan values ... \n")
    imputed_df = impute_values(df=new_df, cols_with_missing=cols_with_missing)
    print(imputed_df.head())

    # Dummy variables
    imputed_df['OST_WEST_KZ'] = pd.get_dummies(imputed_df['OST_WEST_KZ'], prefix='OST_WEST_KZ', drop_first=True)


    print('\n Saving data ... \n')
    total_df = imputed_df.astype('int')
    save_data(total_df, 'total_df.csv')
    print("The dataset was stored!")


    #Store customers id to compare the results
    others_id = np.intersect1d(others_id, df['LNR'].unique())
    customers_id = np.intersect1d(customers_id, df['LNR'].unique())

    np.savetxt('others_id.out', others_id)
    np.savetxt('customers_id.out', customers_id)


if __name__ == '__main__':
    main()
