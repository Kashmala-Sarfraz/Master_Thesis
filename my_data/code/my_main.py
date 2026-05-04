#%% 
# Import functions & set up folder structure
import polars as pl
from pathlib import Path
import pandas as pd
import os
from my_wrds_credentials import get_wrds_credentials

from my_aux_functions import(
    pre_clean_jpn,
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
    eval_strategy_returns,
    eval_strategy_returns_period,
    compute_backtest_metrics,
    plot_cumulative_performance)

BASE_DIR = Path.home()/"Desktop"/"Master_Thesis"/"my_data"
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
setup_folder_structure(data_path=DATA_DIR, base_path=BASE_DIR)
#%% 
# Import raw stock data from WRDS
creds = get_wrds_credentials()
download_stock_characteristics(username=creds.username, password=creds.password, data_path=DATA_DIR)
#%% 
# Create other necessary inputs
nyse_size_cutoffs(data_path=DATA_DIR)
return_cutoffs(data_path=DATA_DIR)
nyse_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"nyse_cutoffs.parquet")
ret_cutoffs = pl.read_parquet(DATA_DIR/"other_input"/"return_cutoffs.parquet")
ret_cutoffs = ret_cutoffs.with_columns((pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1"))
#%% 
# pre_clean_jpn(data_path=DATA_DIR)
#%%
# Create stock level portfolios
lms_ret = {}
for ex in ["USA"]:
    for pfs in [10, 3, 4]:

        key = (ex, pfs)
        print(key)

        lms_ret[key] = factor_returns(
            data_path=DATA_DIR, excntry=ex, nyse_cutoffs_df=nyse_cutoffs, ret_cutoffs_df=ret_cutoffs, bp_min_n=10,pfs=pfs)
#%% 
feature = {}
target = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for adjust in [0, 1, 2, 3]:

            key = (cntry, pfs, adjust)

            print(key)

            feature[key], target[key] = build_factor_characteristics(
                data_path=DATA_DIR, excntry=cntry, pfs=pfs, adj=adjust)

#%% 
# Define train, val and test periods
splits_idx = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for adjust in [0, 1, 2, 3]:

            key = (cntry, pfs, adjust)

            print(key)

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

            splits_idx[key] = splits
#%% Train models
ml_pred = {}
ml_imp = {}

for cntry in ["USA"]:
        for pfs in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, adjust)
            
                print(key)
                
                if key not in splits_idx:
                    continue

                ml_pred[key], ml_imp[key] = train_pred_model(
                    data_path=DATA_DIR, excntry=cntry, pfs=pfs, adj=adjust,
                    splits_idx=splits_idx[key], model= "all")

#%% Global Evaluation
ml_pred_gl = {}
ml_imp_gl = {}

for cntry in ["USA"]:
        for pfs in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, adjust)

                print(key)
            
                ml_pred_gl[key], ml_imp_gl[key] = eval_model(
                     data_path=DATA_DIR, pfs=pfs, excntry=cntry, adj=adjust)
#%% Add COMB
ml_pred_w_comb = {}
ml_pred_gl_w_comb = {}
ml_imp_w_comb = {}
ml_imp_gl_w_comb = {}

for cntry in ["USA"]:
        for pfs in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, adjust)
            
                print(key)
                
                (ml_pred_w_comb[key], ml_pred_gl_w_comb[key],
                 ml_imp_w_comb[key], ml_imp_gl_w_comb[key]) = add_comb_model(
                    data_path=DATA_DIR, pfs=pfs, excntry=cntry, adj=adjust)

#%%
# Build Factor Selection Strategy and Scenario Analysis
buck_ret_avg_gl = {}
buck_ret_avg_mo = {}
buck_ret_ts = {}
buck_ret_per_mo = {}
buck_ret_per_gl = {}

for cntry in ["USA"]:
     for pfs in [10, 3, 4]:
         for n_buck in [10, 3, 4]:
                 for adjust in [0, 1, 2, 3]:
                   
                   key = (cntry, pfs, n_buck, adjust)

                   print(key)

                   (buck_ret_avg_gl[key], buck_ret_avg_mo[key],
                    buck_ret_ts[key], buck_ret_per_mo[key],
                    buck_ret_per_gl[key]) = build_strategy_returns(
                        data_path=DATA_DIR, excntry=cntry, pfs=pfs,n_buckets=n_buck,adj=adjust)

 #%%
# Report Alphas and T stats for the whole Period
regress_strat_gl = {}
strat_turno = {}
regress_buck_gl = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for n_buck in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:
                
                key = (cntry, pfs, n_buck, adjust)

                print(key)

                (regress_strat_gl[key], strat_turno[key],
                regress_buck_gl[key]) = eval_strategy_returns(
                    data_path=DATA_DIR, excntry = cntry, pfs = pfs, n_buckets=n_buck, adj=adjust)

#%%
# Report Alphas and T stats for Crashes
regress_strat_gl_crash = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for n_buck in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, n_buck, adjust)

                print(key)

                regress_strat_gl_crash[key] = eval_strategy_returns_period(
                     data_path=DATA_DIR, excntry = cntry, pfs = pfs, n_buckets=n_buck, adj=adjust)
# %%
# Backtest Metrics
bt_metrics_gl = {}
bt_month = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for n_buck in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, n_buck, adjust)

                print(key)

                bt_month[key], bt_metrics_gl[key] = compute_backtest_metrics(
                     data_path=DATA_DIR, excntry = cntry, pfs = pfs, n_buckets=n_buck, adj=adjust)
# %%

plot_paths = {}

for cntry in ["USA"]:
    for pfs in [10, 3, 4]:
        for n_buck in [10, 3, 4]:
            for adjust in [0, 1, 2, 3]:

                key = (cntry, pfs, n_buck, adjust)

                print(key)

                plot_paths[key] = plot_cumulative_performance(
                    data_path=DATA_DIR,base_path= BASE_DIR, excntry=cntry, pfs=pfs, n_buckets=n_buck,
                    adj=adjust, save=True, show=False
                )

# %%
