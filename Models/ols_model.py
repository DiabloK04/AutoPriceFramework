#%% Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')
#%% Close all previous figures
plt.close("all")
#%% Reproducibility settings
SEED = 125  # seed for reproducibility
r = 0.7     # Set r as fraction of the subset that is going to be used as training set
r = np.clip(r, 0.6, 0.8) # Typically r = 0.6 - 0.8, and making sure that chosen r stays in that range
#%% Functions for cleaning and transforming the dataset
# Convert date to pure year, removing description column and rows where target variable has NaN values
def cleaning(X):
    # Creating a cop to make the changes
    X_clean = X.copy()

    # Converting the seller to binary variable
    X_clean['seller'] = X_clean['seller'].map({'Autobedrijf': 1, 'Particulier': 0})

    # Filling empty entries as No and converting history to binary variable
    X_clean['history'] = X_clean['history'].fillna('Nee')
    X_clean['history'] = X_clean['history'].map({'Ja': 1, 'Nee': 0})
    
    # Renaming the column to warranty_label (for the plots)
    X_clean.rename(columns={'predicted_label':'warranty_label'}, inplace=True)
    
    # Calculating the age of the car
    X_clean['year'] = pd.to_datetime(X_clean['year'].astype(str), format="%b/%y", errors="coerce").dt.year
    X_clean['age'] = 2025 - X_clean['year']

    # Dropping apk and year 
    X_clean = X_clean.drop(columns=['year', 'apk'])
    
    # Dropping observations where price is unknown and remove outliers
    X_clean = X_clean.dropna(subset=['price_euro'])
    X_clean = X_clean[X_clean['price_euro']<1e6 and X_clean['price_euro']>1] # Get rid of price outliers 
    
    return X_clean

# Creating pipelines that handle NaN values in features
def create_pipelines():
    # Pipeline for handling NaN values in categorical features
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe',  OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'))
        ])

    # Pipeline for handling NaN values in nummerical features
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
        ])

    # Pipeline for owners
    owner_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler',StandardScaler())
        ])

    return cat_pipe, num_pipe, owner_pipe
#%%
def develop_OLS_model(data, warranty):
    # Creating the feature matrix
    if warranty == True:
        X = data.drop(columns=['price_euro','make'])
    else:
        X = data.drop(columns=['price_euro','make', 'warranty_label'])
    
    # Defining target variable                           
    y = data['price_euro']
    
    # Categorical features
    cat_feat = ['color','fuel','model','transmission']
    
    # Numerical features
    if warranty == True:
        num_feat = ['mileage', 'power', 'weight', 'age', 'warranty_label']
    else:
        num_feat = ['mileage', 'power', 'weight', 'age']
    
    # The pipelines for NaN values
    cat_pipe, num_pipe, owner_pipe = create_pipelines()
    
    # Transforming the columns
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, cat_feat),
            ('num', num_pipe, num_feat),
            ('owners', owner_pipe, ['owners']),
            ],
        remainder='passthrough'
        )
    
    # Pipeline for preprocessing and fitting linear regression
    ols_pipeline = Pipeline([
        ('preprocess',preprocess),
        ('OLS',LinearRegression())
        ])
    
    # Splitting the data into training/test test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-r, random_state = SEED)
    
    # Fitting linear regression on training set
    ols_pipeline.fit(X_train, y_train)
    
    # Getting the predictions
    y_test_pred = ols_pipeline.predict(X_test)
    
    # Evaluation Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Get the OLS model
    ols_model = ols_pipeline.named_steps['OLS']
    
    # Getting the coefficients and intercept
    coefficients = ols_model.coef_
    intercept = ols_model.intercept_
    
    # Getting feature names from the preprocessing steps
    feat_names = ols_pipeline.named_steps['preprocess'].get_feature_names_out()
    feat_names = [name.split('__')[-1] for name in feat_names]
    
    # Apply preprocessing to X_train
    X_train_transformed = ols_pipeline.named_steps['preprocess'].transform(X_train)
    
    # Calculate residuals: Corrected to use the transformed data
    residuals = y_train - ols_model.predict(X_train_transformed)
    X_train_transformed = np.asarray(X_train_transformed, dtype=np.float64)
    
    # Determine VIF scores
    X_transformed_df = pd.DataFrame(X_train_transformed, columns=feat_names)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_transformed_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_transformed_df.values, i) for i in range(X_transformed_df.shape[1])]
    
    # Calculate variance-covariance matrix
    X_transpose = X_train_transformed.T
    X_transpose_X = np.dot(X_transpose, X_train_transformed)
    
    # Calculate standard errors (diagonal of the covariance matrix)
    var_b = np.linalg.inv(X_transpose_X) * np.var(residuals, ddof=X_train_transformed.shape[0] - X_train_transformed.shape[1])
    se = np.sqrt(np.diagonal(var_b))
    
    # Calculate t-statistics
    t_stat = coefficients / se
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), df=X_train_transformed.shape[0] - X_train_transformed.shape[1])) for i in t_stat]
    
    # Calculate se, t-statistics, p-values for intercept
    intercept_se = np.sqrt(np.var(residuals) * np.linalg.inv(np.dot(X_transpose, X_train_transformed)).diagonal()[0])
    intercept_t_stat = intercept / intercept_se
    intercept_p_value = 2 * (1 - stats.t.cdf(np.abs(intercept_t_stat), df=X_train_transformed.shape[0] - X_train_transformed.shape[1]))
    
    # Storing the results
    results = {
        'vif': vif_data,
        'Feature_names': feat_names,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'Intercept': {'Coefficient': intercept,'P-Value': intercept_p_value}
    }
    
    for name, coef, p_value in zip(feat_names, coefficients, p_values):
        results[name] = {'Coefficient': coef, 'P-Value': p_value}
    
    return results
#%% ✅ Loading the dataset
df = pd.read_csv('clean_final.csv', delimiter=',') # The large dataset
#%% Cleaning the dataset and divide it per brand
df_clean = cleaning(df)
num_brands = df_clean['make'].nunique()

df_brands = {}
brands = []
for s in range(0, num_brands):
    brands.append(df_clean['make'].value_counts().index[s])
    df_brands[brands[s]] = df_clean[(df_clean['make']==brands[s])]
#%% Execute the code for the top ten brands
warranty=True

res = {} # Results of model for top ten brands
for i in range(0,10):
    data = df_brands.get(brands[i])
    results = develop_OLS_model(data, warranty)
    res[brands[i]] = results