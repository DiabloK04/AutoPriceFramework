#%% Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import os
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance, PartialDependenceDisplay, partial_dependence
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')
#%% Close all previous figures and creating folders for all plots
plt.close("all")

os.makedirs("Feature Importance", exist_ok=True)
os.makedirs("LIME", exist_ok=True)
os.makedirs("PDP", exist_ok=True)
#%% Reproducibility settings
SEED = 125  # seed for reproducibility
k = 5       # k for k-fold cross-validation
n_ite = 300 # Number of iterations for RandomizedSearchCV
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
    X_clean = X_clean[X_clean['price_euro']<1e6] # Get rid of price outliers 
    X_clean = X_clean[X_clean['price_euro']>1] # Get rid of price outliers
    
    return X_clean

# Creating pipelines that handle NaN values in features
def create_pipelines():
    # Pipeline for handling NaN values in categorical features
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe',  OneHotEncoder(handle_unknown='ignore',sparse_output=False))
        ])

    # Pipeline for handling NaN values in nummerical features
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ])

    # Pipeline for owners
    owner_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True))
        ])

    return cat_pipe, num_pipe, owner_pipe
#%% XGBoost search grid
def searching_parameters():
    xgbrf_search_grid = {
        'XGBRF__n_estimators': randint(100, 1000),  
        'XGBRF__max_depth': randint(5,50),  
        'XGBRF__subsample': uniform(0.1, 0.9),
        'XGBRF__colsample_bytree': uniform(0.1, 0.9),
        'XGBRF__min_child_weight': randint(1, 10)
        }
    return xgbrf_search_grid
#%% Function for developing XGB Random Forest
def develop_XGBRF(X, y, warranty):
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
            ('owners', owner_pipe, ['owners'])
            ],
        remainder='passthrough'
        )
    
    # Pipeline for preprocessing and fitting to RF
    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('XGBRF', XGBRFRegressor(objective='reg:squarederror', learning_rate=1, random_state = SEED))
        ])
    
    # Splitting the data into training/test test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-r, random_state = SEED)
    
    # Tuning hyperparameters based on training data
    xgbrf_search_grid = searching_parameters()
    random_search = RandomizedSearchCV(pipeline, xgbrf_search_grid, n_iter = n_ite, cv = k, n_jobs = -1, verbose = 0, random_state = SEED, scoring = 'neg_root_mean_squared_error')
    random_search.fit(X_train, y_train)
    
    # Optimal hyperparameters and model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    # Cross-validation
    cv_results = random_search.cv_results_
    cv_neg_rmse_scores = cv_results['mean_test_score'] 
    cv_rmse_scores = -cv_neg_rmse_scores  
    mean_cv_rmse = np.mean(cv_rmse_scores)
    best_cv_rmse = -random_search.best_score_
    
    # Pipeline for transforming the dataset and fitting it to the XGBRF
    model = Pipeline([
        ('preprocess',preprocess),
        ('XGBRF',XGBRFRegressor(
            objective = 'reg:squarederror',
            learning_rate = 1,
            n_estimators = best_params['XGBRF__n_estimators'],
            max_depth = best_params['XGBRF__max_depth'],
            subsample = best_params['XGBRF__subsample'],
            colsample_bytree = best_params['XGBRF__colsample_bytree'],
            min_child_weight = best_params['XGBRF__min_child_weight'],
            random_state = SEED
            ))
        ])
        
    # Retrain the model on training set
    model.fit(X_train, y_train)
    xgbrf_final = model.named_steps['XGBRF']

    # Performing the model on unseen test data
    y_test_predict = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    test_r2 = r2_score(y_test, y_test_predict)

    # Generalization error from cross-validation to test
    general_error = 100 * (test_rmse - mean_cv_rmse) / mean_cv_rmse

    # Feature names after transformation
    feat_names = model.named_steps['preprocess'].get_feature_names_out()
    
    # Variable Importance
    perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=20,
    random_state=SEED,
    scoring='neg_root_mean_squared_error')
    
    # Results
    results = {
        'best_model': best_model,
        'best_params': best_params,
        'mean_cv_rmse': mean_cv_rmse,
        'best_cv_rmse': best_cv_rmse,
        'general_error': general_error,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'final_model': model,
        'xgbrf_final': xgbrf_final,
        'permutation_importance': perm_result,
        'permutation_names': X.columns,
        'feature_names': feat_names,
        'test_data': X_test,
        'train_data': X_train,
        'test_price': y_test
    }
    return results
#%% Results function
def print_results(results):
    best_params = results['best_params']
    
    print('-' * 15, 'Hyper parameters', '-' * 15)                                       
    print(f"Number of trees: {best_params['XGBRF__n_estimators']}")
    print(f"Maximum depth of trees: {best_params['XGBRF__max_depth']}")
    print(f"Subsample ratio of the training instances: {best_params['XGBRF__subsample']}")
    print(f"Subsample ratio of columns when constructing each tree: {best_params['XGBRF__colsample_bytree']}")
    print(f"Minimum sum of instance weight needed in a child: {best_params['XGBRF__min_child_weight']}")
    
    print('-' * 15, 'CV RMSE', '-' * 15)
    print(f"Best CV Root Mean Squared Error: {results['best_cv_rmse']:.2f}")
    print(f"Mean CV Root Mean Squared Error: {results['mean_cv_rmse']:.2f}")

    print('-' * 15, 'Test Performance', '-' * 15)
    print(f"Test RMSE: {results['test_rmse']:.2f}")
    print(f"Test R²: {results['test_r2']:.3f}")

    print('-' * 15, 'Generalization Error', '-' * 15)
    print(f"Generalization Error (CV → Test): {results['general_error']:.2f}%")
#%% Plot Permutation Importance in %RMSE
def plot_feature_importance(results, current_brand):
    perm_names = results['permutation_names']
    perm_result = results['permutation_importance']
    
    perm_mean = -perm_result.importances_mean  # Make it positive (increase in error)
    abs_perm_mean = np.abs(perm_mean)
    perm_proc = 100 * abs_perm_mean / np.sum(abs_perm_mean) 
    
    # Creating DataFrame for the features
    feat_df = pd.DataFrame({
        'Feature': perm_names,
        '%IncRMSE': perm_proc}).sort_values(by='%IncRMSE', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_df,y='Feature',x='%IncRMSE')
    plt.title('Feature Importances XGB Random Forest - '+str(current_brand))
    plt.xlabel('%IncRMSE')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('Feature Importance/'+str(current_brand)+'_features_imp_xgbrf')
    plt.close()
#%% Partial Dependence Plots
def get_partial_dependence(results, current_brand, warranty):
    X_test = results['test_data']
    model = results['final_model']
    xgbrf_final = results['xgbrf_final']
    transformed_X_test = model.named_steps['preprocess'].transform(X_test)
    
    # Cleaning up the feature names
    feat_names = results['feature_names']
    feat_names = [name.split('__')[-1] for name in feat_names]
    transformed_X_test = pd.DataFrame(transformed_X_test, columns=feat_names)
    
    # Features for partial dependence plots
    if warranty == True:
        feat_names = ['mileage', 'power', 'weight', 'age', 'warranty_label']
    else:
        feat_names = ['mileage', 'power', 'weight', 'age']
        
    part_values = {}
    for feat in feat_names:
        pd_values = partial_dependence(xgbrf_final, transformed_X_test, feat)
        grid = pd_values['grid_values'][0]
        partial_dependence_values = pd_values['average'][0]
        part_values[feat] = {'grid': grid, 'partial dependence': partial_dependence_values}
        
        # Plot if it has more than point
        if len(partial_dependence_values) > 1:
            PartialDependenceDisplay.from_estimator(
                xgbrf_final, transformed_X_test, [feat], grid_resolution=50)
        
            plt.title('Partial Dependence Plot XGB Random Forest - '+str(current_brand))
            plt.ylabel('Predicted price (€)')
            plt.grid(True)
            plt.savefig('PDP/'+str(current_brand)+'_pdp_imp_xgbrf_'+str(feat))
            plt.close()
    
    return part_values
#%% LIME
def get_lime(results, current_brand, num_feat):
    xgbrf_final = results['xgbrf_final']
    model = results['best_model']
    X_test = results['test_data']
    X_train = results['train_data']
    feat_names = results['feature_names']
    feat_names = [name.split('__')[-1] for name in feat_names]
    transformed_X_test = model.named_steps['preprocess'].transform(X_test)
    transformed_X_train = model.named_steps['preprocess'].transform(X_train)
    
    # Keeping index within bounds
    num_feat = np.clip(num_feat, 0, transformed_X_test.shape[1])
    
    # Training the explainer    
    explainer = LimeTabularExplainer(
        training_data = transformed_X_train.astype(float),
        feature_names = feat_names,
        mode='regression',
        random_state=SEED,
        discretize_continuous=False
    )

    outliers = {}
    for k in range(0, X_test.shape[0]):
        # Select the observation
        observation = transformed_X_test[k].astype(float)
        
        # Get explanation
        explanation = explainer.explain_instance(
            data_row = observation,
            predict_fn = xgbrf_final.predict,
            num_features=num_feat
        )
        
        # Search for warranty_label in the top ten contributors
        for j in range(0, num_feat):
            if explanation.as_list()[j][0] == 'warranty_label' and j <= 9:
                outliers[k] = j
                break
    
    # Sort the outliers based on their rankings
    sorted_outliers = sorted(outliers.items(), key=lambda item: item[1])
    
    return sorted_outliers 

def plot_lime(results, current_brand, obs_index, num_feat, warranty):
    xgbrf_final = results['xgbrf_final']
    model = results['final_model']
    X_test = results['test_data']
    X_train = results['train_data']
    feat_names = results['feature_names']
    feat_names = [name.split('__')[-1] for name in feat_names]
    y_test_predict = model.predict(X_test)
    y_test = results['test_price']
    transformed_X_test = model.named_steps['preprocess'].transform(X_test)
    transformed_X_train = model.named_steps['preprocess'].transform(X_train)
    
    # Keeping index within bounds
    obs_index = np.clip(obs_index, 0, X_test.shape[0]-1)
    num_feat = np.clip(num_feat, 0, transformed_X_test.shape[1])
        
    explainer = LimeTabularExplainer(
        training_data = transformed_X_train.astype(float),
        feature_names = feat_names,
        mode='regression',
        random_state=SEED,
        discretize_continuous=False
    )
    
    # Select the observation
    observation = transformed_X_test[obs_index].astype(float)
    
    # Get explanation
    explanation = explainer.explain_instance(
        data_row = observation,
        predict_fn = xgbrf_final.predict,
        num_features=num_feat
    )
    
    # Information about observation
    if warranty == True:
        info = ['color','fuel','mileage','model','power','transmission','weight', 'age', 'warranty_label']
    else:
        info = ['color','fuel','mileage','model','power','transmission','weight', 'age']
        
    obs_info = f"Observation {obs_index} from test data \n Brand: {current_brand}\n"
    for feat in info:
        feat_index = results['permutation_names'].get_loc(feat)
        obs_info += f"{feat}: {X_test.iloc[obs_index][feat_index]}\n"
    
    # Visualize
    print(explanation.as_list())
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    plt.ylabel('Feature')
    plt.xlabel('Feature Importance (€)')
    plt.title(f'True Price: €{y_test.iloc[obs_index]:.2f} | Predicted Price: €{y_test_predict[obs_index]:.2f}')
    plt.annotate(obs_info, 
                 xy=(0.36, 0.36), xycoords='axes fraction', 
                 ha='right', va='top', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('LIME/Lime_'+str(current_brand)+'_xgbrf')
    plt.close()
    return explanation
#%% Main function
def main(current_brand, warranty):
    # Get the relevant dataset and their size
    data = df_brands.get(current_brand)
    size = data.shape[0]
    
    # Creating the feature matrix
    if warranty == True:
        X = data.drop(columns=['price_euro','make'])
    else:
        X = data.drop(columns=['price_euro','make','warranty_label'])
    
    # Defining target variable                           
    y = data['price_euro']
    
    # Running time for XGBRF
    start_time = time.time()
    results = develop_XGBRF(X, y, warranty)
    total_time = time.time() - start_time
    
    # Get total_time in hr:min:sec format
    hours, rest = divmod(total_time, 3600)
    minutes, seconds = divmod(rest, 60)
    
    print('-' * 15, 'Results ' + current_brand , '-' * 15)
    print('-' * 15, 'Settings for XGBRF model', '-' * 15)                                      
    print(f"Seed for randomization: {SEED}")
    print(f"k for k-fold CV: {k}")
    print(f"Training/test set as % of used data: {r*100:.2f}%/{(1-r)*100:.2f}%")
    print(f"No. of iterations for RandomizedSearchCV: {n_ite}")
    print(f"Size of the dataset: {size}")
    print_results(results)
    print('-' * 15, 'Total Run Time', '-' * 15)
    print(f"Total elapsed runtime XGB Random Forest: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
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
warranty=True # True if model with BERT warranty, False if not

res = {} # Results of model for top ten brands
out = {} # List where warranty_label is scored in the top ten in LIME
if __name__ == "__main__":
    start_run = time.time()
    for j in range(0, 10):
        current_brand = brands[j]
        results = main(current_brand, warranty)
        res[current_brand] = results
        if warranty==True:
            # Plot feature importance, LIME and PDP plots
            plot_feature_importance(results, current_brand)
            outliers = get_lime(results, current_brand, 15)
            out[current_brand] = outliers
            if not outliers:
                plot_lime(results, current_brand, 5, 15, warranty)
            else:
                plot_lime(results, current_brand, outliers[0][0], 15, warranty)
            part_values = get_partial_dependence(results, current_brand, warranty)
    
    run_time = time.time() - start_run
    hours, rest = divmod(run_time, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"Run time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
#%% Results to csv file
res_summary = pd.DataFrame([
    {
     'Brand': brand,
     'Best CV RMSE': results['best_cv_rmse'],
     'Mean CV RMSE': results['mean_cv_rmse'],
     'Test RMSE': results['test_rmse'],
     'Test R2': results['test_r2'],
     'Generalization Error (CV to Test) %': results['general_error'],
     'n_estimators': results['best_params']['XGBRF__n_estimators'],
     'max_depth': results['best_params']['XGBRF__max_depth'],
     'subsample': results['best_params']['XGBRF__subsample'],
     'colsample_bytree': results['best_params']['XGBRF__colsample_bytree'],
     'min_child_weight': results['best_params']['XGBRF__min_child_weight'],
     'Seed': SEED,
     'k-fold cross-validation': k
     }
    for brand, results in res.items()
    ])

if warranty == True:
    out_summary = []
    for brand, outliers in out.items():
        for obs_index, rank in outliers:
            out_summary.append({
                'Brand': brand,
                'Observation Index': obs_index,
                'Rank of warranty_label': rank
            })
    
    out_df = pd.DataFrame(out_summary)

if warranty is True:
    res_summary.to_csv('xgbrf_model_warranty_results.csv', index=False)
    out_df.to_csv('xgbrf_lime_warranty_outliers.csv', index=False)
else:
    res_summary.to_csv('xgbrf_model_no_warranty_results.csv', index=False)