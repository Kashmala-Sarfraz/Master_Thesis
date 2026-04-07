#%% Scripthead
import polars as pl
from pathlib import Path
import os


from my_aux_functions import(
    setup_folder_structure,
    download_stock_characteristics,
    get_countries,
    nyse_size_cutoffs,
    return_cutoffs,
    factor_returns,
    build_factor_characteristics,
    build_train_val_test_idx,
    predict_with_OLS,
    predict_with_pls)

from my_wrds_credentials import get_wrds_credentials

BASE_DIR = Path.home()/"Desktop"/"Master_Thesis"/"my_data"
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
#%% Get Credentials 
creds = get_wrds_credentials()
setup_folder_structure(data_path=DATA_DIR)
#%%
download_stock_characteristics(username=creds.username, password=creds.password, data_path=DATA_DIR)
#%%
countries = get_countries(data_path=DATA_DIR)
nyse_size_cutoffs(data_path=DATA_DIR)
return_cutoffs(data_path=DATA_DIR)
nyse_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"nyse_cutoffs.parquet")
ret_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"return_cutoffs.parquet")
ret_cutoffs = ret_cutoffs.with_columns((pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1"))
#%%
for ex in countries:
    print(ex)
    factor_returns(
        data_path=DATA_DIR,
        excntry=ex,
        nyse_cutoffs_df=nyse_cutoffs,
        ret_cutoffs_df=ret_cutoffs,
        bp_min_n=10,
        pfs=3
    )
#%%
for ex in countries:
    print(ex)

    build_factor_characteristics(data_path=DATA_DIR, excntry=ex, pfs=3)
# %%
splits_idx = {}

for ex in countries:
    print(ex)
    
    splits = build_train_val_test_idx(
        data_path=DATA_DIR,
        excntry=ex,
        min_train_val = 15,
        val = 5,
        test_range = 1,
        forward_steps = 1)
    
    if not splits: 
        continue
    
    splits_idx[ex] = splits
#%%
for ex in countries:
    print(ex)

    if ex in splits_idx:
        out_ols = predict_with_OLS(
            excntry=ex,
            data_path=DATA_DIR,
            pfs= 3,
            splits_idx=splits_idx[ex])
        
        out_pls = predict_with_pls(
            excntry=ex,
            data_path=DATA_DIR,
            pfs= 3,
            splits_idx=splits_idx[ex])

    







        






# X_numerical = X.select_dtypes(include="number").copy()
# display(X_numerical.describe(include = "all").T)


# %%

# %%
