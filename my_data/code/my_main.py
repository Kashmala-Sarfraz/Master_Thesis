#%% 
# Import functions & set up folder structure
import polars as pl
from pathlib import Path
import pandas as pd
import os
from my_wrds_credentials import get_wrds_credentials

from my_aux_functions import(
    setup_folder_structure,
    download_stock_characteristics,
    nyse_size_cutoffs,
    return_cutoffs,
    factor_returns,
    build_factor_characteristics,
    build_train_val_test_idx,
    train_pred_model,
    add_comb_model,
    eval_model,
    build_strategy_returns,
    eval_strategy_returns)

BASE_DIR = Path.home()/"Desktop"/"Master_Thesis"/"my_data"
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
setup_folder_structure(data_path=DATA_DIR)
#%% 
# Import raw stock data from WRDS
# creds = get_wrds_credentials()
# download_stock_characteristics(username=creds.username, password=creds.password, data_path=DATA_DIR)
#%% 
# Create other necessary inputs
nyse_size_cutoffs(data_path=DATA_DIR)
return_cutoffs(data_path=DATA_DIR)
nyse_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"nyse_cutoffs.parquet")
ret_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"return_cutoffs.parquet")
ret_cutoffs = ret_cutoffs.with_columns((pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1"))
#%% 
# Create stock level portfolios
for ex in ["USA"]:
    for pfs in [3]: #, 4, 10
        print(ex)
        factor_returns(
            data_path=DATA_DIR,
            excntry=ex,
            nyse_cutoffs_df=nyse_cutoffs,
            ret_cutoffs_df=ret_cutoffs,
            bp_min_n=10,
            pfs=pfs)
#%% 
# Build feature and target      
for ex in ["USA"]:
    for pfs in [3]: #, 4, 10
        for adjust in [0, 1]:
            build_factor_characteristics(
            data_path=DATA_DIR, excntry=ex, pfs=pfs, adj=adjust)

#%% 
# Define train, val and test periods
splits_idx = {}
for cntry in ["USA"]:  # , "JPN"
    for pfs in [3]:  # , 4, 10
        for adjust in [0, 1]:

            print(f"ADJUSTMENT: {adjust}")

            splits = build_train_val_test_idx(
                data_path=DATA_DIR,
                pfs=pfs,
                adj=adjust,
                excntry=cntry,
                min_train_val=15,
                val=5,
                test_range=1,
                forward_steps=1
            )

            if not splits:
                continue

            splits_idx[(cntry, pfs, adjust)] = splits
#%% Train models and evaluate performance

for cntry in ["USA"]: #, "JPN"
        for pfs in [3]: #, 4, 10
            for adjust in [0, 1]:
            
                print(cntry)
                
                if (cntry, pfs, adjust) not in splits_idx:
                    continue

                r2_split, y_pred_split, y_true_split, test_idx_split, vi_split = train_pred_model(
                    data_path=DATA_DIR,
                    excntry=cntry,
                    pfs=pfs,
                    splits_idx=splits_idx[cntry, pfs, adjust],
                    model= "all", 
                    adj=adjust)
                
                eval_model(
                    data_path=DATA_DIR,
                    pfs=pfs,
                    excntry=cntry,
                    vi_split=vi_split,
                    r2_split=r2_split,
                    y_pred_split=y_pred_split,
                    y_true_split=y_true_split,
                    test_idx_split=test_idx_split,
                    adj=adjust)
                
                monthly_df, global_df = add_comb_model(data_path=DATA_DIR, pfs=pfs, excntry=cntry, adj=adjust)

#%%
for cntry in ["USA"]:
     for pfs in [3]:
         for n_buck in [10]:
                 for adjust in [0,1]:
                    global_ret_df = build_strategy_returns(
                     data_path=DATA_DIR,
                     excntry = cntry,
                     pfs = pfs,
                     n_buckets=n_buck,
                     adj=adjust
                     )

 #%%
for cntry in ["USA"]:
    for pfs in [3]:
        for n_buck in [10]:
            for adjust in [0, 1]:
                eval_strategy_returns(
                    data_path=DATA_DIR,
                    excntry = cntry,
                    pfs = pfs,
                    n_buckets=n_buck,
                    adj=adjust)



#%%