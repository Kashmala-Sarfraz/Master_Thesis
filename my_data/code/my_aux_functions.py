import time
import datetime
from datetime import date
from pathlib import Path
import os
import duckdb
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sktime.split import ExpandingWindowSplitter
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np

# the 153 characteristics
chars = [
    "age",
    "aliq_at",
    "aliq_mat",
    "ami_126d",
    "at_be",
    "at_gr1",
    "at_me",
    "at_turnover",
    "be_gr1a",
    "be_me",
    "beta_60m",
    "beta_dimson_21d",
    "betabab_1260d",
    "betadown_252d",
    "bev_mev",
    "bidaskhl_21d",
    "capex_abn",
    "capx_gr1",
    "capx_gr2",
    "capx_gr3",
    "cash_at",
    "chcsho_12m",
    "coa_gr1a",
    "col_gr1a",
    "cop_at",
    "cop_atl1",
    "corr_1260d",
    "coskew_21d",
    "cowc_gr1a",
    "dbnetis_at",
    "debt_gr3",
    "debt_me",
    "dgp_dsale",
    "div12m_me",
    "dolvol_126d",
    "dolvol_var_126d",
    "dsale_dinv",
    "dsale_drec",
    "dsale_dsga",
    "earnings_variability",
    "ebit_bev",
    "ebit_sale",
    "ebitda_mev",
    "emp_gr1",
    "eq_dur",
    "eqnetis_at",
    "eqnpo_12m",
    "eqnpo_me",
    "eqpo_me",
    "f_score",
    "fcf_me",
    "fnl_gr1a",
    "gp_at",
    "gp_atl1",
    "ival_me",
    "inv_gr1",
    "inv_gr1a",
    "iskew_capm_21d",
    "iskew_ff3_21d",
    "iskew_hxz4_21d",
    "ivol_capm_21d",
    "ivol_capm_252d",
    "ivol_ff3_21d",
    "ivol_hxz4_21d",
    "kz_index",
    "lnoa_gr1a",
    "lti_gr1a",
    "market_equity",
    "mispricing_mgmt",
    "mispricing_perf",
    "ncoa_gr1a",
    "ncol_gr1a",
    "netdebt_me",
    "netis_at",
    "nfna_gr1a",
    "ni_ar1",
    "ni_be",
    "ni_inc8q",
    "ni_ivol",
    "ni_me",
    "niq_at",
    "niq_at_chg1",
    "niq_be",
    "niq_be_chg1",
    "niq_su",
    "nncoa_gr1a",
    "noa_at",
    "noa_gr1a",
    "o_score",
    "oaccruals_at",
    "oaccruals_ni",
    "ocf_at",
    "ocf_at_chg1",
    "ocf_me",
    "ocfq_saleq_std",
    "op_at",
    "op_atl1",
    "ope_be",
    "ope_bel1",
    "opex_at",
    "pi_nix",
    "ppeinv_gr1a",
    "prc",
    "prc_highprc_252d",
    "qmj",
    "qmj_growth",
    "qmj_prof",
    "qmj_safety",
    "rd_me",
    "rd_sale",
    "rd5_at",
    "resff3_12_1",
    "resff3_6_1",
    "ret_1_0",
    "ret_12_1",
    "ret_12_7",
    "ret_3_1",
    "ret_6_1",
    "ret_60_12",
    "ret_9_1",
    "rmax1_21d",
    "rmax5_21d",
    "rmax5_rvol_21d",
    "rskew_21d",
    "rvol_21d",
    "sale_bev",
    "sale_emp_gr1",
    "sale_gr1",
    "sale_gr3",
    "sale_me",
    "saleq_gr1",
    "saleq_su",
    "seas_1_1an",
    "seas_1_1na",
    "seas_11_15an",
    "seas_11_15na",
    "seas_16_20an",
    "seas_16_20na",
    "seas_2_5an",
    "seas_2_5na",
    "seas_6_10an",
    "seas_6_10na",
    "sti_gr1a",
    "taccruals_at",
    "taccruals_ni",
    "tangibility",
    "tax_gr1a",
    "turnover_126d",
    "turnover_var_126d",
    "z_score",
    "zero_trades_126d",
    "zero_trades_21d",
    "zero_trades_252d",
]


def measure_time(func):
    """
    Description:
        decorator to measure the execution time of a function in minutes:seconds. 
        Decorators are functions that take another function as an argument 
        and extend its behavior without explicitly modifying it.
    Steps:
        1) Record start time and print function name + start.
        2) Execute the wrapped function and capture result.
        3) Record end time; compute and print duration.
        4) Return the original result.

    Output:
        Prints timing info to stdout, which means the decorator displays timing messages to the console
        instead of returning them as data; 
        but it does return wrapped function's result.
    """
    # we dont know how many arguments the wrapped function will take, so we use *args (positional)
    # and we dont know how the keyword arguments will look like, so we use **kwargs (keyword)
    # important: args and kwrgs are just conventions,
    # we could name them differently but this is the standard way to do it
    def wrapper(*args, **kwargs):
        # time.time() returns the current time in seconds since the epoch (January 1, 1970, 00:00:00 UTC)
        # Output looks like: 1770723101.8555498
        start_time = time.time()
        print(f"Function: {func.__name__.upper()}", flush = True)
        # localtime converts seconds to date structure.
        # Output looks like: time.struct_time(tm_year=2026, tm_mon=2, tm_mday=10, tm_hour=12, tm_min=31, tm_sec=41, tm_wday=1, tm_yday=41, tm_isdst=0)
        # strftime formats the time in a human-readable way
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", flush = True)
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush = True)
        duration = end_time - start_time
        # this is floor division it keeps the integer and removes the decimel
        # so 24.758982133865356 is turned to 24.0
        minutes = duration // 60
        # this gives us the remainder after floor division
        seconds = duration % 60 
        # .:2f is a formatting syntax inside a f string
        print(f"Execution time: {minutes} minutes and {seconds:.2f} seconds")
        return result
    return wrapper

def setup_folder_structure(data_path, base_path):
    """
    Description:
        Create the projects folder structure before dowloading data

    Steps: 
        1. Define current path
        2. Get parent directory
        3. Create a folder "data" and its subfolders: "factor_returns", "stock_characteristics" and "portfolio_returns"
    
    Output: no output
    """

    (data_path/"factor_returns").mkdir(parents=True, exist_ok=True)
    (data_path/"stock_characteristics").mkdir(parents=True, exist_ok=True)
    (data_path/"portfolio_returns").mkdir(parents=True, exist_ok=True)
    (data_path/"other_input").mkdir(parents=True, exist_ok=True)
    (data_path/"factor_characteristics").mkdir(parents=True, exist_ok=True)
    (data_path/"ml_model_output").mkdir(parents=True, exist_ok=True)
    (base_path/"exhibits").mkdir(parents=True, exist_ok=True)

def gen_wrds_connection_info(username, password):
    return (
        f"host=wrds-pgdata.wharton.upenn.edu "
        f"port=9737 "
        f"dbname=wrds "
        f"user={username} "
        f"password={password} "
        f"sslmode=require"
    )

def build_projection():

    """
    Description:
    Steps:
    Output: 
    """
    columns = (
        [
            "id",
            "eom",
            "source_crsp",
            "comp_exchg",
            "crsp_exchcd",
            "size_grp",
            "ret_exc",
            "ret_exc_lead1m",
            "me",
            "gics",
            "ff49",
            "excntry",
        ]
            + chars
            )

    
    return ", ".join(columns)

def year_country_map(data_path, wrds_conn, duckdb_conn, lib, table):

    file_path = data_path / "other_input" / "year_country_map.parquet"

    if file_path.exists():
        df = pl.read_parquet(file_path)

    else:
        df = duckdb_conn.execute(
            f"""
            SELECT excntry, MIN(YEAR(date)) AS min_year
            FROM postgres_scan('{wrds_conn}', '{lib}', '{table}')
            GROUP BY excntry"""
            ).pl()
        
        df.write_parquet(file_path)

    year_map = dict(zip(
        df["excntry"].cast(pl.Utf8),
        df["min_year"].cast(pl.Int64)))

    return year_map


def download_ff_monthly(data_path, wrds_conn, duckdb_conn, lib, table):

    file_path = data_path/"other_input"/"ff_monthly.parquet"

    if file_path.exists():
        print(f"file exists under: {file_path}")
    else:
        print(f"downloading ff under: {file_path}")
        duckdb_conn.execute(
            f"""
            COPY (SELECT
                    (date_trunc('month', dateff) + interval '1 month' - interval '1 day') AS eom,
                    CAST(mktrf AS DOUBLE) AS mkt,
                    CAST(smb AS DOUBLE) AS smb,
                    CAST(hml AS DOUBLE) AS hml,
                    CAST(rmw AS DOUBLE) AS rmw,
                    CAST(cma AS DOUBLE) AS cma,
                    CAST(umd AS DOUBLE) AS wml
                    FROM postgres_scan('{wrds_conn}', '{lib}', '{table}')) 
                    TO '{str(file_path)}' (FORMAT PARQUET); 
                        """)

def download_query_wrds(
        wrds_conn, duckdb_conn, projection, lib, table, file_name, country, year):
    """
    Description:
        This is where we actually load the data via a sql query and create a parquet file.
    """
    duckdb_conn.execute(f""" 
    COPY (
            SELECT {projection}
            FROM postgres_scan('{wrds_conn}', '{lib}', '{table}')
            WHERE excntry = '{country}'
            AND common=1
            AND exch_main=1
            AND primary_sec=1
            AND obs_main=1
            AND eom >= DATE '{year}-01-31'
            AND eom < DATE '{year+1}-01-31'
                        ) 
    TO '{str(file_name)}' (FORMAT PARQUET); 
    """)
    
@measure_time
def download_stock_characteristics(username, password, data_path):
    """
    Description:
        This is a function that brings all the functions together that are needed
        to download data
    Steps:
        1) Call function that gets WRDS connection info 
        2) Set up duckdb 
        3) Call function that dowloads data from WRDS
    Output:
        doesn't return anything, but at the end the data will be downloaded
    """
    wrds_connection_data = gen_wrds_connection_info(username, password)
    
    # Create DuckDB connection. In general there are two ways to connect.
    # First a temporary database. No file is created. Everything disappears when closed.
    # -> "duckdb.connect(":memory:")"
    # Second a persistant database file. Creates a file on disk. Database survives after Pythons exits.
    # -> "duckdb.connect("my_database.duckdb")"
    # Funfact: Normally SQL is used on databases.
    # But with DuckDB you can local files behave like a database table. Without importing them to a database.
    con = duckdb.connect(":memory:")
    # RAM control (safe for 16GB system)
    con.execute("PRAGMA memory_limit='8GB';")

    con.execute("PRAGMA preserve_insertion_order=false;")
    # Limit disk spill safely
    con.execute("PRAGMA max_temp_directory_size='30GiB';")
    # We need a PostgreSQL extension in duckDB since WRDS is a Postgres datatable
    # and we need this extension to interact with the database
    con.execute("INSTALL postgres; LOAD postgres;")
    
    table_name = "contrib.global_factor"
    lib, table = table_name.split(".") 
    # a projection is the list of columns that you choose in a SELECT statement

    download_ff_monthly(
        data_path = data_path,
        wrds_conn=wrds_connection_data,
        duckdb_conn=con,
        lib="ff_all",
        table="fivefactors_monthly")


    countries = ['USA'] #, 'JPN', 'BRA', 'CHN', 'GBR', 'DEU', 'IND'

    year_country_dict = year_country_map(
        data_path = data_path,
        wrds_conn=wrds_connection_data,
        duckdb_conn=con,
        lib=lib,
        table=table)
    
    projection = build_projection()

    try:
        for country in countries:

            start = year_country_dict[country]
            
            for year in range(start, 2026):

                print(f"data for {country} in {year}", flush=True)
                file_path = data_path / "stock_characteristics" / f"{country}_{year}.parquet"     

                if file_path.exists():
                    continue
            
                try:
                    download_query_wrds(
                        wrds_conn=wrds_connection_data,
                        duckdb_conn=con,
                        lib=lib,
                        table=table,
                        projection = projection,
                        country=country,
                        year=year,
                        file_name=file_path
                        )
                    
                except Exception as e:
                    print(f"failed {country}--{year}: {e}")
                    if file_path.exists():
                        file_path.unlink()
                    raise

    finally:
        con.close()


def nyse_size_cutoffs(data_path):

    path = data_path / "stock_characteristics"
    files = list(path.glob("USA_*.parquet"))

    nyse_sf = pl.scan_parquet(files).sql(
        """
        SELECT 
        eom,
        COUNT(*) as n,
        QUANTILE_DISC(me, 0.01) as nyse_p1,
        QUANTILE_DISC(me, 0.20) as nyse_p20,
        QUANTILE_DISC(me, 0.50) as nyse_p50,
        QUANTILE_DISC(me, 0.80) as nyse_p80

        FROM self
        WHERE crsp_exchcd = 1
        AND me IS NOT NULL

        GROUP BY eom
        ORDER BY eom 
        """
    )

    nyse_sf.sink_parquet(data_path/"other_input"/"nyse_cutoffs.parquet")

def return_cutoffs(data_path):

    path = data_path/"stock_characteristics"
    files = list(path.glob("USA_*.parquet"))

    data = pl.scan_parquet(files).filter(pl.col("source_crsp") == 1)

    data = data.sql(
        """ 
        SELECT 
        eom,
        COUNT(ret_exc) AS n,
        QUANTILE_DISC(ret_exc, 0.001) AS p001,
        QUANTILE_DISC(ret_exc, 0.999) AS p999

        FROM self
        GROUP BY eom
        ORDER BY eom

        """)
    
    data.sink_parquet(data_path/"other_input"/"return_cutoffs.parquet")


def add_ecdf(df: pl.DataFrame) -> pl.DataFrame:
    # How many stocks take each value of the signal in each month? f.ex 3 stocks have p/e ratio of 4
    count = df.filter(pl.col("bp_stock")).group_by(["eom", "var"]).agg(n_ref=pl.len())
    cdf_values = (count.sort(["eom", "var"])
                  .with_columns(cdf_val=(pl.cum_sum("n_ref")/pl.sum("n_ref")).over("eom"))
                  .select(["eom", "var", "cdf_val"]))
    
    right = cdf_values.sort(["eom", "var"])
    left = df.sort(["eom", "var"])

    # for each value (f.ex p/e ratio) in the left table,
    # we look backward along the right table
    # to find the closest value that does not exceed it
    # Summary pick largest value on the right that is smaller than the value on the left

    out = (left
              .join_asof(right, on="var", by="eom", strategy="backward")
              .with_columns(pl.col("cdf_val").fill_null(0.0).alias("cdf"))
              .drop("cdf_val"))

    return out


char_info = (
    pl.read_excel(
        "https://github.com/bkelly-lab/ReplicationCrisis/raw/master/GlobalFactors/Factor%20Details.xlsx",
        sheet_name="details",
    )
    .filter(pl.col("abr_jkp").is_not_null())
    .select([pl.col("abr_jkp").alias("characteristics"), pl.col("direction").cast(pl.Int32)])
)

# def pre_clean_jpn(data_path):

#     files = list((data_path / "stock_characteristics").glob(f"JPN_*.parquet"))
#     temp = []
#     for f in tqdm(files, desc="Processing files"):
        
#         data = pl.read_parquet(f)
        
#         exclude = ["id", "eom", "source_crsp", "size_grp", "excntry"]
#         data = data.with_columns([
#             (pl.col(i).cast(pl.Float64)) for i in data.columns if i not in exclude
#             ])
        
#         temp.append(data)
        
#     df = pl.concat(temp)
    
#     min_available_share = 0.5

#     drop_cols = [
#         c for c in df.columns
#         if df.schema[c].is_float()
#         and df.select(
#             (~pl.col(c).is_nan() & pl.col(c).is_not_null()).mean()
#         ).item() < min_available_share
#     ]
#     keep_cols = [c for c in data.columns if c not in drop_cols]

#     out_dir = data_path / "stock_characteristics_clean"
#     out_dir.mkdir(exist_ok=True)
    
#     for f in files:
#         data = pl.read_parquet(f)
#         data.select(keep_cols).write_parquet(out_dir / f.name)
     
def factor_returns(
        data_path,
        excntry,
        nyse_cutoffs_df,
        ret_cutoffs_df,
        bp_min_n,
        pfs
        ):
    
    files = list((data_path / "stock_characteristics").glob(f"{excntry}_*.parquet"))

    temp_files = []
    temp_market_files = []
    
    for f in tqdm(files, desc="Processing files"):
        
        data = pl.read_parquet(f)
                
        exclude = ["id", "eom", "source_crsp", "size_grp", "excntry"]
        data = data.with_columns([
            (pl.col(i).cast(pl.Float64)) for i in data.columns if i not in exclude
            ])

        data = data.filter(
            (pl.col("size_grp").is_not_null()) &
            (pl.col("me").is_not_null()) &
            (pl.col("ret_exc_lead1m").is_not_null())) 

        data = data.join(nyse_cutoffs_df.select(["eom", "nyse_p80"]), on="eom", how="left")
        data = data.with_columns(
            pl.min_horizontal(pl.col("nyse_p80"), pl.col("me")).alias("me_cap")
            ).drop("nyse_p80")
        
        data = data.join(
            ret_cutoffs_df.select(["eom_lag1", "p999", "p001"]
                                ).rename({"eom_lag1":"eom"}), on="eom", how="left")
        data = data.with_columns(
            pl.when(pl.col("source_crsp") == 0)
            .then(
                pl.when(pl.col("ret_exc_lead1m") < pl.col("p001"))
                .then(pl.col("p001"))
                .when(pl.col("ret_exc_lead1m") > pl.col("p999"))
                .then(pl.col("p999"))
                .otherwise(pl.col("ret_exc_lead1m"))
                )
            .otherwise(pl.col("ret_exc_lead1m"))
            .alias("ret_exc_lead1m")
        ).drop(["source_crsp", "p001", "p999"])


        market_partial = (
            data
            .select(["eom", "ret_exc_lead1m", "me_cap"])
            .with_columns(
                (pl.col("ret_exc_lead1m") * pl.col("me_cap")).alias("ret_x_me")
            )
            .group_by("eom")
            .agg([
                pl.col("ret_x_me").sum().alias("sum_ret_x_me"),
                pl.col("me_cap").sum().alias("sum_me_cap")
            ])
        )

        market_path = data_path / "factor_returns" / f"temp_market_{f.stem}.parquet"
        market_partial.write_parquet(market_path)
        temp_market_files.append(market_path)

        data = data.with_columns([
            pl.col(c).clip(
                lower_bound=pl.col(c).quantile(0.01).over("eom"),
                upper_bound=pl.col(c).quantile(0.99).over("eom")
            ).alias(c)
            for c in chars])
        
        for _i, x in enumerate(tqdm(chars, desc="Processing chars", unit="char", ncols=80)):
            sub = (
                data
                .with_columns(pl.col(x).cast(pl.Float64).alias("var"))
                .filter(pl.col("var").is_not_null())
                .select(["eom", "var", "size_grp", "ret_exc_lead1m", "me_cap"] + chars))
            
            # exclude micro 
            sub = sub.with_columns(pl.col("size_grp")
                                .is_in(["mega", "large", "small"])
                                .alias("bp_stock"))
            # min companies in month
            sub = sub.with_columns(bp_n=pl.sum("bp_stock").over("eom")).filter(
                pl.col("bp_n") >= bp_min_n)

            # create buckets
            sub = add_ecdf(sub)

            sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("cdf_min"))
            sub = sub.with_columns(
                pl.when(pl.col("cdf") == pl.col("cdf_min"))
                .then(0.00000001)
                .otherwise(pl.col("cdf"))
                .alias("cdf")
            )
            # turn into n portfolios
            sub = sub.with_columns(
                (pl.col("cdf")*pfs)
                .ceil()
                .clip(lower_bound=1, upper_bound=pfs)
                .alias("pf")
            )

            # value wighted average characteristic and return in each portfolio
            pf_returns = sub.group_by(["pf", "eom"]).agg(
                [
                    pl.lit(x).alias("characteristics"),
                    pl.col("me_cap").sum().alias("sum_me_cap"),
                    pl.len().alias("n"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("ret_exc_lead1m_vw_cap")
                ]
                +
                [
                    ((pl.col(c) * pl.col("me_cap")).filter(pl.col(c).is_not_null()).sum()
                        / pl.col("me_cap").filter(pl.col(c).is_not_null()).sum()).alias(f"{c}_vw_cap")
                        for c in chars
                ]
            )
            
            out_path = data_path / "factor_returns" / f"temp_pf_{excntry}_{x}_{f.stem}.parquet"
            pf_returns.write_parquet(out_path)
            temp_files.append(out_path)
            
        
    pf_returns_total = pl.concat([pl.read_parquet(f) for f in temp_files])
    pf_returns_total = pf_returns_total.with_columns(
        pl.lit(excntry).str.to_uppercase().alias("excntry"))
    
        
    hml_returns = pf_returns_total.group_by(["eom", "characteristics", "excntry"]).agg(
        [# check if there are rows with missing long or short portfolios
        pl.col("pf").is_in([pfs, 1]).sum().alias("pfs"),
        # calculate long short return
        (pl.col("ret_exc_lead1m_vw_cap").filter(pl.col("pf") == pfs).first()
        - pl.col("ret_exc_lead1m_vw_cap").filter(pl.col("pf") == 1).first()).alias("ret_exc_lead1m_vw_cap"),
        # calculate number of stocks in both portfolios combined
        (pl.col("n").filter(pl.col("pf") == pfs).first()
        + pl.col("n").filter(pl.col("pf") == 1).first()).alias("n_stocks"),
        # calculate the min numeber of stock in either of the two portfolios
        (pl.col("n").filter(pl.col("pf").is_in([pfs, 1])).min().alias("n_stocks_min"))]
        + 
        [(
        pl.col(f"{c}_vw_cap").filter(pl.col("pf") == pfs).first()
        -pl.col(f"{c}_vw_cap").filter(pl.col("pf") == 1).first()).alias(f"spread_{c}") for c in chars]
        )
    
    hml_returns = hml_returns.filter(pl.col("pfs") == 2).drop("pfs")
    hml_returns = hml_returns.sort(["characteristics", "eom"])

    lms_returns = char_info.join(hml_returns, on="characteristics", how = "left")
    resign_cols = ["ret_exc_lead1m_vw_cap"] + [f"spread_{c}" for c in chars]
    lms_returns = lms_returns.with_columns([
            (pl.col(var) * pl.col("direction")).alias(var) for var in resign_cols])
    
    market = (
        pl.scan_parquet(data_path / "factor_returns" / f"temp_market_{excntry}*.parquet")
        .group_by("eom")
        .agg([
            pl.col("sum_ret_x_me").sum(),
            pl.col("sum_me_cap").sum()
        ])
        .with_columns(
            (pl.col("sum_ret_x_me") / pl.col("sum_me_cap"))
            .alias("market_ret_exc_vw_cap")
        )
        .select(["eom", "market_ret_exc_vw_cap"])
    )

    market = market.with_columns(pl.col("eom").dt.offset_by("1m").dt.month_end().alias("eom"))

    lms_returns.write_parquet(
        data_path / "factor_returns" / f"{excntry}_{pfs}_lms_returns.parquet")
    hml_returns.write_parquet(
        data_path / "factor_returns" / f"{excntry}_{pfs}_hml_returns.parquet")
    pf_returns_total.write_parquet(
        data_path / "factor_returns" / f"{excntry}_{pfs}_pf_returns.parquet")
    market.sink_parquet(
        data_path / "factor_returns" / f"{excntry}_market_returns.parquet")
    
    for temp_file in temp_files + temp_market_files: 
        temp_file.unlink(missing_ok=True)

    return  lms_returns


def build_factor_characteristics(data_path, pfs, excntry, adj):

    factor = data_path / "factor_returns" / f"{excntry}_{pfs}_lms_returns.parquet"
    market = data_path / "factor_returns" / f"{excntry}_market_returns.parquet"

    if factor.exists() and market.exists():

        lms_returns = pl.read_parquet(factor)
        market_returns = pl.read_parquet(market)

        left = (lms_returns
                .select(["eom", "characteristics", "ret_exc_lead1m_vw_cap", "excntry"])
                .sort(["characteristics", "eom"]))

        df = left.join(market_returns, on="eom", how="left")
        df = df.sort(["characteristics", "eom"])

        if adj == 1:

            df = df.with_columns(
                (pl.col("ret_exc_lead1m_vw_cap")
                 - pl.col("ret_exc_lead1m_vw_cap").mean().over("eom")
                ).alias("ret_exc_cross_lead1m_vw_cap"))

            base_ret = "ret_exc_cross_lead1m_vw_cap"
            cols_to_drop = [base_ret,
                            "ret_exc_lead1m_vw_cap",
                            "excntry",
                            "characteristics",
                            "eom",
                            "market_ret_exc_vw_cap"]
            
            out_X = data_path / "factor_characteristics" / f"{excntry}_{pfs}_feature_cross.parquet"
            out_y = data_path / "factor_characteristics" / f"{excntry}_{pfs}_target_cross.parquet"
            meta_out = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta_cross.parquet"
        
        elif adj == 2:

            df = df.with_columns(
                (pl.col("ret_exc_lead1m_vw_cap")
                - pl.col("ret_exc_lead1m_vw_cap").mean().over("eom")
                ).alias("ret_exc_cross_lead1m_vw_cap")
            )
            
            df = df.with_columns(
                pl.when(pl.col("ret_exc_cross_lead1m_vw_cap") > 0)
                .then(1)
                .when(pl.col("ret_exc_cross_lead1m_vw_cap") <= 0)
                .then(-1)
                .otherwise(None)
                .alias("target_class")
            )

            base_ret = "ret_exc_cross_lead1m_vw_cap"

            cols_to_drop = [
                base_ret,
                "target_class",  
                "ret_exc_lead1m_vw_cap",
                "excntry",
                "characteristics",
                "eom",
                "market_ret_exc_vw_cap"
            ]

            out_X = data_path / "factor_characteristics" / f"{excntry}_{pfs}_feature_class.parquet"
            out_y = data_path / "factor_characteristics" / f"{excntry}_{pfs}_target_class.parquet"
            meta_out = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta_class.parquet"

        elif adj == 3:

            base_ret =  "ret_exc_lead1m_vw_cap"

            cols_to_drop = [
                base_ret,
                "target_rank",
                "excntry",
                "characteristics",
                "eom",
                "market_ret_exc_vw_cap"
            ]

            out_X = data_path / "factor_characteristics" / f"{excntry}_{pfs}_feature_rank.parquet"
            out_y = data_path / "factor_characteristics" / f"{excntry}_{pfs}_target_rank.parquet"
            meta_out = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta_rank.parquet"

        else:
            base_ret = "ret_exc_lead1m_vw_cap"
            cols_to_drop = [base_ret, 
                            "excntry",
                            "characteristics",
                            "eom",
                            "market_ret_exc_vw_cap"]
            
            out_X = data_path / "factor_characteristics" / f"{excntry}_{pfs}_feature.parquet"
            out_y = data_path / "factor_characteristics" / f"{excntry}_{pfs}_target.parquet"
            meta_out = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta.parquet"
        
        df = df.sort(["characteristics", "eom"])
        
        df = df.with_columns([
            pl.col(base_ret)
            .shift(t)
            .over("characteristics")
            .alias(f"ret_m_{t}")
            for t in range(1, 61)
        ])

        df = df.with_columns(
            (
                ((1 + pl.col(base_ret).shift(t*12 + 1)).log())
                .rolling_sum(window_size=12)
                .over("characteristics")
                .exp() - 1
            ).alias(f"ret_y_{t}")
            for t in range(6, 21)
        )

        df = df.with_columns(
            pl.col(base_ret)
            .shift(1)
            .rolling_std(window_size=12*t)
            .over("characteristics")
            .alias(f"vol_m_{t}")
            for t in [2, 3, 4, 5]
        )
        
        df = df.with_columns([
            (
                pl.when(
                    pl.col("market_ret_exc_vw_cap")
                    .rolling_var(window_size=t)
                    .over("characteristics") > 0
                )
                .then(
                    (
                        ((pl.col(base_ret).shift(1) *
                          pl.col("market_ret_exc_vw_cap"))
                         .rolling_mean(window_size=t)
                         .over("characteristics"))
                        -
                        (
                            pl.col(base_ret).shift(1)
                            .rolling_mean(window_size=t)
                            .over("characteristics")
                            *
                            pl.col("market_ret_exc_vw_cap")
                            .rolling_mean(window_size=t)
                            .over("characteristics")
                        )
                    )
                    /
                    pl.col("market_ret_exc_vw_cap")
                    .rolling_var(window_size=t)
                    .over("characteristics")
                )
                .otherwise(None)
            ).alias(f"beta_{t}m")
            for t in [12, 24, 36, 48, 60]
        ])

        df = (df.with_columns(
            pl.col(base_ret).shift(1)
            .cum_count().over("characteristics").alias("n"))
            .with_columns(
                ((
                    (
                        (pl.col("market_ret_exc_vw_cap") *
                         pl.col(base_ret).shift(1))
                        .cum_sum().over("characteristics") / pl.col("n")
                    )
                    -
                    (
                        (pl.col("market_ret_exc_vw_cap")
                         .cum_sum().over("characteristics") / pl.col("n"))
                        *
                        (pl.col(base_ret).shift(1)
                         .cum_sum().over("characteristics") / pl.col("n"))
                    )
                )
                /
                (
                    (pl.col("market_ret_exc_vw_cap") ** 2)
                    .cum_sum().over("characteristics") / pl.col("n")
                    -
                    (pl.col("market_ret_exc_vw_cap")
                     .cum_sum().over("characteristics") / pl.col("n")) ** 2
                )
            ).alias("beta_full"))
        ).drop("n")

        spread_cols = [c for c in lms_returns.columns if c.startswith("spread_")]

        spreads = lms_returns.select(["eom", "characteristics"] + spread_cols)

        df = df.join(spreads, on=["eom", "characteristics"], how="left")

        feature_clip_cols = [
            c for c in df.columns if c.startswith(("ret_m_", "ret_y_", "vol_m_", "beta_"))]
        
        df = df.with_columns([
            pl.col(c).clip(
                lower_bound=pl.col(c).quantile(0.01).over("eom"),
                upper_bound=pl.col(c).quantile(0.99).over("eom")
                ).alias(c)
                for c in feature_clip_cols
                ])
        
        
        to_clean_cols = [
            c for c in df.columns
            if c.startswith(("ret_m_", "ret_y_", "vol_m_", "beta_", "spread_"))
        ]

        target_clean_col = ("target_class" if adj == 2 else base_ret)

        df = df.filter(
            pl.col(target_clean_col).is_not_null() &
            pl.all_horizontal([
                pl.col(c).is_not_null() & pl.col(c).is_finite()
                for c in to_clean_cols
            ])
        )

        if adj == 3:
            df = df.with_columns(
                (
                    2 * (
                        (pl.col("ret_exc_lead1m_vw_cap").rank(method="average").over("eom") - 1)
                        / (pl.len().over("eom") - 1)
                    ) - 1
                ).alias("target_rank")
            )
                        
        df = df.sort(["eom", "characteristics"])

        df = df.with_columns(pl.arange(0, pl.len()).alias("row_id"))

        meta = df.select(["row_id", "eom", "characteristics", "excntry"]).sort("row_id")

        X = df.drop(cols_to_drop).sort("row_id")

        if adj == 1:
             y = df.select(["row_id", base_ret, "ret_exc_lead1m_vw_cap"]).sort("row_id")
        elif adj == 2:
            y = df.select(["row_id", base_ret, "target_class", "ret_exc_lead1m_vw_cap"]).sort("row_id")
        elif adj == 3:
            y = df.select(["row_id", base_ret, "target_rank"]).sort("row_id")
        else:
            y = df.select(["row_id", base_ret]).sort("row_id")

        X.write_parquet(out_X)
        y.write_parquet(out_y)
        meta.write_parquet(meta_out)

        return X, y

def get_suffix(adj):
    if adj == 1:
        return "_cross"
    elif adj == 2:
        return "_class"
    elif adj == 3:
        return "_rank"
    else:
        return ""

def build_train_val_test_idx(data_path, pfs, adj, excntry, min_train_val, val, test_range, forward_steps):

    test = (test_range * 12)
    
    suffix = get_suffix(adj)

    meta = pl.read_parquet(
        data_path/"factor_characteristics"/f"{excntry}_{pfs}_meta{suffix}.parquet").to_pandas()
    
    unique_eom = meta["eom"].drop_duplicates().sort_values().reset_index(drop=True)

    min_req_range = (min_train_val + val + test_range) * 12

    if len(unique_eom) < min_req_range:
        return []

    expand_wind_splits = []

    splitter = ExpandingWindowSplitter(
        fh=list(range(1, test + 1)),
        initial_window=min_train_val * 12,
        step_length=forward_steps * 12)

    for train_val_months_idx, test_months_idx in splitter.split(unique_eom):

        train_months_idx = train_val_months_idx[:-(val*12)]
        val_months_idx = train_val_months_idx[-(val*12):]

        train_months = unique_eom.iloc[train_months_idx]
        val_months = unique_eom.iloc[val_months_idx]
        test_months = unique_eom.iloc[test_months_idx]

        print(f"Train: {train_months.min()} to {train_months.max()} | {len(train_months)} months")
        print(f"Val:   {val_months.min()} to {val_months.max()} | {len(val_months)} months")
        print(f"Test:  {test_months.min()} to {test_months.max()} | {len(test_months)} months")


        train_rows = meta.loc[meta["eom"].isin(train_months), "row_id"].values
        val_rows = meta.loc[meta["eom"].isin(val_months), "row_id"].values
        test_rows = meta.loc[meta["eom"].isin(test_months), "row_id"].values
        

        expand_wind_splits.append({
            "train" : train_rows,
            "val" : val_rows,
            "test" : test_rows
        })
        
    return expand_wind_splits

def get_pos_class_proba(model, X):
    proba = model.predict_proba(X)

    if hasattr(model, "classes_"):
        pos_idx = np.where(model.classes_ == 1)[0][0]
        return proba[:, pos_idx]

    return proba[:, 1]

def predict_with_ols(X_train, y_train, X_test, y_test,  y_pred_split, y_true_split):

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"ols - r2_pred: {r2}")

    y_pred_split.setdefault("OLS", []).append(y_pred.ravel())
    y_true_split.setdefault("OLS", []).append(y_test.values.ravel())

    return model

def predict_with_logit_cls(X_train_val, y_train_val, X_test, y_test,
                            y_pred_split, y_true_split, y_prob_split):

    model = LogisticRegression(
        penalty=None,
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    )

    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    y_prob = get_pos_class_proba(model, X_test)

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"logit_cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("LOGIT_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("LOGIT_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("LOGIT_CLS", []).append(y_prob)

    return model

def tune_model_with_val(model, param_dist, X_train, y_train, X_val, y_val, n_iter):
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))    ])

    ps = PredefinedSplit(test_fold)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="r2",
        n_jobs=-1,
        cv=ps,
        random_state=42,
        refit=False
    )

    search.fit(X_train_val, y_train_val)

    return search.best_params_

def tune_classifier_with_val(model, param_dist, X_train, y_train, X_val, y_val, n_iter):
    
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))
    ])

    ps = PredefinedSplit(test_fold)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        n_jobs=-1,
        cv=ps,
        random_state=42,
        refit=False
    )

    search.fit(X_train_val, y_train_val)

    return search.best_params_

def predict_with_pls(X_train, y_train,
                     X_val, y_val, X_train_val,
                     y_train_val, X_test, y_test,
                     y_pred_split, y_true_split):
                     
    param_dist = {"n_components": list(range(1, 15))}
    
    best_params = tune_model_with_val(
        PLSRegression(max_iter=5000), param_dist, X_train, y_train, X_val, y_val, n_iter=30)
    
    print(f"PLS best params: {best_params}")

    best_model = PLSRegression(max_iter=5000, **best_params)
    
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"pls - r2_pred: {r2}")

    y_pred_split.setdefault("PLS", []).append(y_pred.ravel())
    y_true_split.setdefault("PLS", []).append(y_test.values.ravel())

    return best_model

def predict_with_lasso(X_train,y_train,
                       X_val, y_val,
                       X_train_val, y_train_val,
                       X_test, y_test,
                       y_pred_split, y_true_split):


    param_dist = {"alpha": np.logspace(-5, 0, 150)}

    best_params = tune_model_with_val(
            Lasso(max_iter=5000), param_dist, X_train, y_train, X_val, y_val, n_iter=30)
    print(f"Lasso best params: {best_params}")

    best_model = Lasso(max_iter=5000, **best_params)
    best_model.fit(X_train_val, y_train_val)

    n_relax = 0
    max_relax = 10

    while np.sum(np.abs(best_model.coef_) > 1e-12) < 2 and n_relax < max_relax:
        best_params["alpha"] = best_params["alpha"] / 2
        n_relax += 1
        best_model = Lasso(max_iter=5000, **best_params)
        best_model.fit(X_train_val, y_train_val)

    y_pred = best_model.predict(X_test)

    n_total = len(best_model.coef_)
    n_zero = (best_model.coef_ == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    r2 = r2_score(y_test, y_pred)
    print(f"lasso - r2_pred: {r2}")

    y_pred_split.setdefault("LASSO", []).append(y_pred.ravel())
    y_true_split.setdefault("LASSO", []).append(y_test.values.ravel())

    return best_model

def predict_with_lasso_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    param_dist = {"C": np.logspace(-4, 4, 150)}


    best_params = tune_classifier_with_val(
        LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, random_state=42, n_jobs=-1),
        param_dist,
        X_train,
        y_train,
        X_val,
        y_val,
        n_iter=30
    )

    print(f"LASSO_CLS best params: {best_params}")

    best_model = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000, random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X_train_val, y_train_val)

    n_relax = 0
    max_relax = 10

    while np.sum(np.abs(best_model.coef_) > 1e-12) < 2 and n_relax < max_relax:
        best_params["C"] = best_params["C"] * 2
        n_relax += 1
        best_model = LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, random_state=42, n_jobs=-1, **best_params)
        best_model.fit(X_train_val, y_train_val)

    y_pred = best_model.predict(X_test)
    y_prob = get_pos_class_proba(best_model, X_test)

    n_total = len(best_model.coef_.ravel())
    n_zero = (best_model.coef_.ravel() == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"lasso-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("LASSO_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("LASSO_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("LASSO_CLS", []).append(y_prob)

    return best_model

def predict_with_enet(X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,
                      y_pred_split, y_true_split):

    param_dist = {"alpha": np.logspace(-5, 0, 150)}

    best_params = tune_model_with_val(
            ElasticNet(max_iter=5000, l1_ratio=0.5),
            param_dist, X_train, y_train, X_val, y_val, n_iter=30)

    print(f"ENET best params: {best_params}")

    best_model = ElasticNet(max_iter=5000, l1_ratio=0.5, **best_params)
    best_model.fit(X_train_val, y_train_val)

    n_relax = 0
    max_relax = 10

    while np.sum(np.abs(best_model.coef_) > 1e-12) < 2 and n_relax < max_relax:
        best_params["alpha"] = best_params["alpha"] / 2
        n_relax +=1
        best_model = ElasticNet(max_iter=5000, l1_ratio=0.5, **best_params)
        best_model.fit(X_train_val, y_train_val)

    y_pred = best_model.predict(X_test)

    n_total = len(best_model.coef_)
    n_zero = (best_model.coef_ == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    r2 = r2_score(y_test, y_pred)
    print(f"enet - r2_pred: {r2}")

    y_pred_split.setdefault("ENET", []).append(y_pred.ravel())
    y_true_split.setdefault("ENET", []).append(y_test.values.ravel())

    return best_model

def predict_with_enet_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    param_dist = {"C": np.logspace(-4, 4, 150), "l1_ratio": [0.25, 0.5, 0.75]}

    best_params = tune_classifier_with_val(
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            random_state=42,
            n_jobs=-1
        ),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=30
    )

    print(f"ENET_CLS best params: {best_params}")

    best_model = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=5000,
        random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X_train_val, y_train_val)

    n_relax = 0
    max_relax = 10

    while np.sum(np.abs(best_model.coef_) > 1e-12) < 2 and n_relax < max_relax:
        best_params["C"] = best_params["C"] * 2
        n_relax += 1
        best_model = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000,
            random_state=42, n_jobs=-1, **best_params)
        best_model.fit(X_train_val, y_train_val)

    y_pred = best_model.predict(X_test)
    y_prob = get_pos_class_proba(best_model, X_test)

    n_total = len(best_model.coef_.ravel())
    n_zero = (best_model.coef_.ravel() == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"enet-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("ENET_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("ENET_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("ENET_CLS", []).append(y_prob)

    return best_model


def predict_with_rf(X_train, y_train, X_val, y_val, X_train_val, y_train_val,
                    X_test, y_test,  y_pred_split, y_true_split):

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 0.8],
    }

    best_params = tune_model_with_val(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, X_train, y_train, X_val, y_val, n_iter=30)

    print(f"RF best params: {best_params}")

    best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"rf - r2_pred: {r2}")

    y_pred_split.setdefault("RF", []).append(y_pred.ravel())
    y_true_split.setdefault("RF", []).append(y_test.values.ravel())

    return best_model


def predict_with_rf_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", 0.3, 0.5, 0.8],
    }

    best_params = tune_classifier_with_val(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=30
    )

    print(f"RF_CLS best params: {best_params}")

    best_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        **best_params
    )

    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)
    y_prob = get_pos_class_proba(best_model, X_test)

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"rf-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("RF_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("RF_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("RF_CLS", []).append(y_prob)

    return best_model

def predict_with_gbrt(
        X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,
        y_pred_split, y_true_split):
    
    param_dist = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [10, 20, 50],
        "max_iter": [100, 200]
    }

    best_params = tune_model_with_val(
        HistGradientBoostingRegressor(random_state=42),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=30
        
    )
    print(f"GBRT best params: {best_params}")

    best_model =  HistGradientBoostingRegressor(random_state=42, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"gbrt - r2_pred: {r2}")

    y_pred_split.setdefault("GBRT", []).append(y_pred.ravel())
    y_true_split.setdefault("GBRT", []).append(y_test.values.ravel())

    return best_model


def predict_with_gbrt_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    param_dist = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [10, 20, 50],
        "max_iter": [100, 200]
    }

    best_params = tune_classifier_with_val(
        HistGradientBoostingClassifier(random_state=42),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=30
    )

    print(f"GBRT-cls best params: {best_params}")

    best_model = HistGradientBoostingClassifier(random_state=42, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)
    y_prob = get_pos_class_proba(best_model, X_test)

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"gbrt-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("GBRT_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("GBRT_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("GBRT_CLS", []).append(y_prob)

    return best_model


def predict_with_xgb(
    X_train, y_train,
    X_val, y_val,
    X_train_val, y_train_val,
    X_test, y_test,
     y_pred_split, y_true_split):
    
    best_score = -np.inf
    best_params = None
    best_iteration = None

    for lr in [0.05, 0.1]:
        for max_d in [3, 5, 7]:
            for subs in [0.6, 0.8]:
                for colsample in [0.7, 0.9]:
                    for reg_lambda in [1, 5]:
                        for min_child_weight in [1, 5]:


                            model = XGBRegressor(
                                learning_rate=lr,
                                n_estimators=1000,
                                max_depth=max_d,
                                subsample=subs,
                                colsample_bytree=colsample,
                                reg_lambda=reg_lambda,
                                min_child_weight=min_child_weight,
                                objective="reg:squarederror",
                                early_stopping_rounds=20,
                                random_state=42,
                                n_jobs=-1
                            )

                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )

                            y_pred_val = model.predict(X_val)
                            metric = r2_score(y_val, y_pred_val)

                            if metric > best_score:
                                best_score = metric
                                best_params = {
                                    "learning_rate": lr,
                                    "max_depth": max_d,
                                    "subsample": subs,
                                    "colsample_bytree": colsample,
                                    "reg_lambda": reg_lambda,
                                    "min_child_weight": min_child_weight
                                }
                                best_iteration = model.best_iteration

    print(f"XGB best params: {best_params}")
    print(f"XGB best score: {best_score}")
    print(f"XGB best iteration: {best_iteration}")

    best_model = XGBRegressor(
        **best_params,
        n_estimators=best_iteration + 1 if best_iteration is not None else 100, 
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train_val, y_train_val, verbose=False)

    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"xgb - r2_pred: {r2}")

    y_pred_split.setdefault("XGB", []).append(y_pred.ravel())
    y_true_split.setdefault("XGB", []).append(y_test.values.ravel())
    

    return best_model


def predict_with_xgb_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    y_train_xgb = np.where(np.asarray(y_train) == 1, 1, 0)
    y_val_xgb = np.where(np.asarray(y_val) == 1, 1, 0)
    y_train_val_xgb = np.where(np.asarray(y_train_val) == 1, 1, 0)
    y_test_xgb = np.where(np.asarray(y_test) == 1, 1, 0)

    best_score = -np.inf
    best_params = None
    best_iteration = None

    for lr in [0.05, 0.1]:
        for max_d in [3, 5, 7]:
            for subs in [0.6, 0.8]:
                for colsample in [0.7, 0.9]:
                    for reg_lambda in [1, 5]:
                        for min_child_weight in [1, 5]:


                            model = XGBClassifier(
                                learning_rate=lr,
                                n_estimators=1000,
                                max_depth=max_d,
                                subsample=subs,
                                colsample_bytree=colsample,
                                reg_lambda=reg_lambda,
                                min_child_weight=min_child_weight,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                early_stopping_rounds=20,
                                random_state=42,
                                n_jobs=-1
                            )

                            model.fit(
                                X_train,
                                y_train_xgb,
                                eval_set=[(X_val, y_val_xgb)],
                                verbose=False
                            )

                            y_pred_val = model.predict(X_val)
                            metric = balanced_accuracy_score(y_val_xgb, y_pred_val)

                            if metric > best_score:
                                best_score = metric
                                best_params = {
                                    "learning_rate": lr,
                                    "max_depth": max_d,
                                    "subsample": subs,
                                    "colsample_bytree": colsample,
                                    "reg_lambda": reg_lambda,
                                    "min_child_weight": min_child_weight
                                }
                                best_iteration = model.best_iteration

    print(f"XGB_CLS best params: {best_params}")
    print(f"XGB_CLS best score: {best_score}")
    print(f"XGB_CLS best iteration: {best_iteration}")

    best_model = XGBClassifier(
        **best_params,
        n_estimators=best_iteration + 1 if best_iteration is not None else 100,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train_val, y_train_val_xgb, verbose=False)

    y_pred_xgb = best_model.predict(X_test)

    y_pred = np.where(y_pred_xgb == 1, 1, -1)
    y_prob = get_pos_class_proba(best_model, X_test)

    acc = balanced_accuracy_score(y_test, y_pred)

    print(f"xgb-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("XGB_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("XGB_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("XGB_CLS", []).append(y_prob)

    return best_model

def predict_with_ffnn(X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,  y_pred_split, y_true_split):
    
    param_dist = {
        "learning_rate_init": [0.0001, 0.0003, 0.001],
        "alpha": [0.0001, 0.001, 0.01, 0.05],
        "batch_size": [32, 64, 128]
    }

    best_params = tune_model_with_val(
        MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=42),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=20,
        
    )

    print(f"FFNN best params: {best_params}")

    best_model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        **best_params)
    
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"ffnn - r2_pred: {r2}")

    y_pred_split.setdefault("FFNN", []).append(y_pred.ravel())
    y_true_split.setdefault("FFNN", []).append(y_test.values.ravel())
    
    return best_model

def predict_with_ffnn_cls(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        y_pred_split, y_true_split, y_prob_split):

    param_dist = {
        "learning_rate_init": [0.0001, 0.0003, 0.001],
        "alpha": [0.0001, 0.001, 0.01, 0.05],
        "batch_size": [32, 64, 128]
    }

    best_params = tune_classifier_with_val(
        MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=42
        ),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=20
    )

    print(f"FFNN_CLS best params: {best_params}")

    best_model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42,
        **best_params
    )

    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)
    y_prob = get_pos_class_proba(best_model, X_test)

    acc = balanced_accuracy_score(y_test, y_pred)
    print(f"ffnn-cls - balanced_accuracy: {acc}")

    y_pred_split.setdefault("FFNN_CLS", []).append(y_pred.ravel())
    y_true_split.setdefault("FFNN_CLS", []).append(y_test.values.ravel())
    y_prob_split.setdefault("FFNN_CLS", []).append(y_prob)

    return best_model

def save_train_pred_outputs(data_path, excntry, pfs, adj,
                            y_pred_split, y_true_split, y_prob_split,
                            test_idx_split, y_ret):

    suffix = get_suffix(adj)

    base = data_path / "ml_model_output"
    pred_path = base / f"{excntry}_{pfs}_pred_month{suffix}.parquet"

    pred_lst = []

    row_ids_all = np.concatenate(test_idx_split)

    for model in y_pred_split.keys():

        y_pred_all = np.concatenate(y_pred_split[model])
        y_true_all = np.concatenate(y_true_split[model])
        
        out = {
            "row_id": row_ids_all,
            "model": model,
            "prediction": y_pred_all,
            "actual": y_true_all,
            "country": excntry,
            "pfs": pfs
        }

        if y_prob_split is not None and model in y_prob_split:
            out["probability"] = np.concatenate(y_prob_split[model])
        else:
            out["probability"] = np.full(len(y_pred_all), np.nan)

        pred_lst.append(
        pl.DataFrame(out).with_columns([
            pl.col("prediction").cast(pl.Float64),
            pl.col("actual").cast(pl.Float64),
            pl.col("probability").cast(pl.Float64),
        ])
    )

    df_pred = pl.concat(pred_lst, how="diagonal")
    df_pred = df_pred.join(y_ret, on="row_id", how="left")
    
    if pred_path.exists():

        existing = pl.read_parquet(pred_path).with_columns([
            pl.col("prediction").cast(pl.Float64),
            pl.col("actual").cast(pl.Float64),
            pl.col("probability").cast(pl.Float64)])
        
        df_pred = pl.concat([existing, df_pred], how="diagonal")
        df_pred = df_pred.unique(subset=["row_id", "model", "country", "pfs"], keep="last")

    df_pred.write_parquet(pred_path)

    return df_pred

def save_vi_output(data_path, excntry, pfs, adj, vi_split, test_idx_split, feature_names):

    suffix = get_suffix(adj)
    base = data_path / "ml_model_output"

    vi_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"
    meta = (
        pl.read_parquet(data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet")
        .select(["row_id", "eom"]))

    vi_lst = []

    for model, vi_list in vi_split.items():

        for i, vi_arr in enumerate(vi_list):

            row_ids = test_idx_split[i]

            test_dates = (
                meta
                .filter(pl.col("row_id").is_in(row_ids))
                .select("eom")
            )

            test_period = test_dates["eom"].min()

            vi_lst.append(
                pl.DataFrame({
                    "test_start": test_period,
                    "model": model,
                    "feature": feature_names,
                    "importance": vi_arr,
                    "country": excntry,
                    "pfs": pfs
                }).with_columns([pl.col("importance").cast(pl.Float64)])
            )

    df_vi = pl.concat(vi_lst, how="diagonal")

    df_vi = (
        df_vi
        .with_columns(
            pl.when(pl.col("importance") < 0)
            .then(0)
            .otherwise(pl.col("importance"))
            .alias("importance")
        )
        .with_columns(
            (
                pl.col("importance")
                / pl.col("importance").sum().over(["test_start", "model", "country", "pfs"])
            )
            .fill_nan(0)
            .fill_null(0)
            .alias("importance")
        )
    )

    if vi_path.exists():
        
        existing = pl.read_parquet(vi_path).with_columns(
            [pl.col("importance").cast(pl.Float64)])

        df_vi = pl.concat([existing, df_vi], how="diagonal")
        df_vi = df_vi.unique(subset=["test_start", "model", "feature", "country", "pfs"], keep="last")

    df_vi.write_parquet(vi_path)

    return df_vi


def train_pred_model(data_path, excntry, pfs, splits_idx, model, adj):

    if adj == 1:
        suffix = "_cross"
        target_col = "ret_exc_cross_lead1m_vw_cap"
    elif adj == 2:
        suffix = "_class"
        target_col = "target_class"
    elif adj == 3:
        suffix = "_rank"
        target_col = "target_rank"
    else:
        suffix = ""
        target_col = "ret_exc_lead1m_vw_cap"

    base = data_path/"factor_characteristics"

    X_path = base /f"{excntry}_{pfs}_feature{suffix}.parquet"
    y_path = base /f"{excntry}_{pfs}_target{suffix}.parquet"

    if adj == 2:
        y_ret = (
            pl.read_parquet(y_path)
            .rename({
                "ret_exc_lead1m_vw_cap": "ret_raw",
                "ret_exc_cross_lead1m_vw_cap": "ret_cross"
            })
            .select(["row_id", "ret_raw", "ret_cross"]))
        
    elif adj == 3:
        y_ret = (
            pl.read_parquet(y_path)
            .rename({"ret_exc_lead1m_vw_cap": "ret_raw"})
            .select(["row_id", "ret_raw"]))
    else:
        y_ret = (
            pl.read_parquet(y_path)
            .rename({"ret_exc_lead1m_vw_cap": "ret_raw"})
            .select(["row_id", "ret_raw"]))

    X_raw = pl.read_parquet(X_path).to_pandas()
    y_raw = pl.read_parquet(y_path).to_pandas()

    X = X_raw.set_index("row_id")
    y = y_raw.set_index("row_id")

    feature_names = X.columns.to_list()

    y_series = y[target_col]

    y_pred_split = {}
    y_true_split = {}
    y_prob_split = {}
    vi_split = {}
    test_idx_split = []

    if adj == 2:
        if model == "all":
            models_to_run = ["LOGIT_CLS", "LASSO_CLS", "ENET_CLS", "RF_CLS", "GBRT_CLS", "XGB_CLS"] #, "FFNN_CLS"
        elif isinstance(model, str):
            models_to_run = [model.upper()]
        else:
            models_to_run = [m.upper() for m in model]
    else:
        if model == "all":
            models_to_run = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB"] #, "FFNN"
        elif isinstance(model, str):
            models_to_run = [model.upper()]
        else:
            models_to_run = [m.upper() for m in model]
            
    n_splits = len(splits_idx)
    
    for split_no, split in enumerate(splits_idx, start=1):

        print("\n" + "=" * 60)
        print(f"SPLIT {split_no}/{n_splits}")
        print("=" * 60)

        train_index = split["train"]
        val_index = split["val"]
        test_index = split["test"]
        train_val_index = np.concatenate([train_index, val_index])

        X_train_val = X.loc[train_val_index]
        X_train = X.loc[train_index]
        X_val = X.loc[val_index]
        X_test = X.loc[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        X_train_val = scaler.transform(X_train_val)

        y_train = y_series.loc[train_index]
        y_val = y_series.loc[val_index]
        y_train_val = y_series.loc[train_val_index]
        y_test = y_series.loc[test_index]

        test_idx_split.append(test_index)
        
        if adj == 2:

            if "LOGIT_CLS" in models_to_run:
                model_logit = predict_with_logit_cls(
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                result = permutation_importance(
                    model_logit, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy"
                )

                vi_split.setdefault("LOGIT_CLS", []).append(result.importances_mean)


            if "LASSO_CLS" in models_to_run:
                model_lasso_logit = predict_with_lasso_cls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                result = permutation_importance(
                    model_lasso_logit, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy"
                )

                vi_split.setdefault("LASSO_CLS", []).append(result.importances_mean)


            if "ENET_CLS" in models_to_run:
                model_enet_logit = predict_with_enet_cls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                result = permutation_importance(
                    model_enet_logit, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy")

                vi_split.setdefault("ENET_CLS", []).append(result.importances_mean)


            if "RF_CLS" in models_to_run:
                model_rf_cls = predict_with_rf_cls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                result = permutation_importance(
                    model_rf_cls, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy")

                vi_split.setdefault("RF_CLS", []).append(result.importances_mean)


            if "GBRT_CLS" in models_to_run:
                model_gbrt_cls = predict_with_gbrt_cls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                result = permutation_importance(
                    model_gbrt_cls, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy")

                vi_split.setdefault("GBRT_CLS", []).append(result.importances_mean)


            if "XGB_CLS" in models_to_run:
                model_xgb_cls = predict_with_xgb_cls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
                )

                y_test_xgb = np.where(np.asarray(y_test) == 1, 1, 0)
                result = permutation_importance(
                    model_xgb_cls, X_test, y_test_xgb, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy")

                vi_split.setdefault("XGB_CLS", []).append(result.importances_mean)


            # if "FFNN_CLS" in models_to_run:
            #     model_ffnn_cls = predict_with_ffnn_cls(
            #         X_train=X_train, y_train=y_train,
            #         X_val=X_val, y_val=y_val,
            #         X_train_val=X_train_val, y_train_val=y_train_val,
            #         X_test=X_test, y_test=y_test,
            #         y_pred_split=y_pred_split, y_true_split=y_true_split, y_prob_split=y_prob_split
            #     )

            #     result = permutation_importance(
            #         model_ffnn_cls, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42, scoring="balanced_accuracy")

            #     vi_split.setdefault("FFNN_CLS", []).append(result.importances_mean)

        else:
            
            if "OLS" in models_to_run:
                
                model_ols = predict_with_ols(
                X_train=X_train_val, y_train=y_train_val,
                X_test=X_test, y_test=y_test,
                y_pred_split=y_pred_split, y_true_split=y_true_split)

                result = permutation_importance(
                    model_ols, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)
                
                vi_split.setdefault("OLS", []).append(result.importances_mean)


            if "PLS" in models_to_run:

                model_pls = predict_with_pls(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                     y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_pls, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

                vi_split.setdefault("PLS", []).append(result.importances_mean)


            if "LASSO" in models_to_run:

                model_lasso = predict_with_lasso(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                     y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_lasso, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

                vi_split.setdefault("LASSO", []).append(result.importances_mean)


            if "ENET" in models_to_run:

                model_enet = predict_with_enet(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                     y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_enet, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)
                
                vi_split.setdefault("ENET", []).append(result.importances_mean)

            if "RF" in models_to_run:

                model_rf = predict_with_rf(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                     y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_rf, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

                vi_split.setdefault("RF", []).append(result.importances_mean)


            if "GBRT" in models_to_run:

                model_gbrt = predict_with_gbrt(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_gbrt, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

                vi_split.setdefault("GBRT", []).append(result.importances_mean)


            # if "FFNN" in models_to_run:

            #     model_ffnn = predict_with_ffnn(
            #         X_train=X_train, y_train=y_train,
            #         X_val=X_val, y_val=y_val,
            #         X_train_val=X_train_val, y_train_val=y_train_val,
            #         X_test=X_test, y_test=y_test,
            #         y_pred_split=y_pred_split, y_true_split=y_true_split
            #     )

            #     result = permutation_importance(
            #         model_ffnn, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            #     vi_split.setdefault("FFNN", []).append(result.importances_mean)


            if "XGB" in models_to_run:

                model_xgb = predict_with_xgb(
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_train_val=X_train_val, y_train_val=y_train_val,
                    X_test=X_test, y_test=y_test,
                    y_pred_split=y_pred_split, y_true_split=y_true_split
                )

                result = permutation_importance(
                    model_xgb, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

                vi_split.setdefault("XGB", []).append(result.importances_mean)

    df_pred = save_train_pred_outputs(data_path, excntry, pfs, adj,
                                      y_pred_split=y_pred_split, y_true_split=y_true_split,
                                      test_idx_split=test_idx_split, y_ret=y_ret,
                                      y_prob_split=y_prob_split if adj == 2 else None)
    
    df_vi = save_vi_output(data_path, excntry, pfs, adj,
                   vi_split=vi_split, test_idx_split=test_idx_split, feature_names=feature_names)
    
    return df_pred, df_vi


def eval_correlation(df, excntry, pfs, adj, data_path):
    
    suffix = get_suffix(adj)
        
    meta = pl.read_parquet(
        data_path/"factor_characteristics"/f"{excntry}_{pfs}_meta{suffix}.parquet").to_pandas()
    
    df = df.merge(meta[["row_id", "eom"]], on="row_id", how="left")
    pear_lst = []
    spear_lst = []

    for _, month in df.groupby("eom"):

        if adj == 2:
            x = month["probability"]
            y = month["ret_cross"]

        else:
            x = month["prediction"]
            y = month["actual"]
        
        if x.std() == 0 or y.std() == 0:
            print(month)
            continue

        pear_lst.append(x.corr(y))
        spear_lst.append(x.corr(y, method = 'spearman'))
    
    pearson = np.nanmean(pear_lst)
    spearman = np.nanmean(spear_lst)
        
    return {"pearson": pearson, "spearman": spearman}


def eval_regression(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "balanced_accuracy": None,}

def eval_classification(y_true, y_pred):
    return {
        "r2": None,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}

def eval_model(data_path, pfs, excntry, adj):

    suffix = get_suffix(adj)

    base = data_path / "ml_model_output"

    pred_month_path = base / f"{excntry}_{pfs}_pred_month{suffix}.parquet"
    vi_month_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"

    df_monthly = pl.read_parquet(pred_month_path)

    if adj == 2:
        df_monthly = df_monthly.with_columns([
            pl.col("prediction").cast(pl.Int64),
            pl.col("actual").cast(pl.Int64),
            pl.col("probability").cast(pl.Float64),
            pl.col("ret_raw").cast(pl.Float64),
            pl.col("ret_cross").cast(pl.Float64),
        ])
    else:
        df_monthly = df_monthly.with_columns([
            pl.col("prediction").cast(pl.Float64),
            pl.col("actual").cast(pl.Float64),
            pl.col("ret_raw").cast(pl.Float64),
        ])

    results_global_lst = []

    for model in df_monthly["model"].unique().to_list():

        model_df = df_monthly.filter(pl.col("model") == model)

        y_true_all = model_df["actual"].to_numpy()
        y_pred_all = model_df["prediction"].to_numpy()

        out_corr = eval_correlation(df=model_df.to_pandas(), excntry=excntry, pfs=pfs, adj=adj, data_path=data_path)
        
        out_score = (
            eval_classification(y_true_all.astype(int), y_pred_all.astype(int)) if adj == 2
            else eval_regression(y_true_all, y_pred_all))
   
        results_global_lst.append({
            "country": excntry,
            "pfs": pfs,
            "model": model,
            "r2": out_score["r2"],
            "balanced_accuracy": out_score["balanced_accuracy"],
            "spearman": out_corr["spearman"],
            "pearson": out_corr["pearson"]
        })

    df_global = (
        pl.from_pandas(pd.DataFrame(results_global_lst))
        .with_columns([
            pl.col("r2").cast(pl.Float64),
            pl.col("balanced_accuracy").cast(pl.Float64),
            pl.col("spearman").cast(pl.Float64),
            pl.col("pearson").cast(pl.Float64),
        ])
    )
    
    results_global_path = base / f"{excntry}_{pfs}_eval_global{suffix}.parquet"

    if results_global_path.exists():
        existing = pl.read_parquet(results_global_path)
        df_global = pl.concat([existing, df_global], how="diagonal")
        df_global = df_global.unique(subset=["model", "country", "pfs"], keep="last")

    df_global.write_parquet(results_global_path)
    print(df_global)

    if vi_month_path.exists():

        df_vi_month = pl.read_parquet(vi_month_path)

        df_vi_global = (
            df_vi_month
            .group_by(["feature", "model", "country", "pfs"])
            .agg(pl.col("importance").mean().alias("importance"))
            .with_columns(
                (
                    pl.col("importance")
                    / pl.col("importance").sum().over(["model", "country", "pfs"])
                )
                .fill_nan(0)
                .fill_null(0)
                .alias("importance")
            )
        )

        vi_global_path = base / f"{excntry}_{pfs}_vi_global{suffix}.parquet"

        if vi_global_path.exists():
            existing = pl.read_parquet(vi_global_path)
            df_vi_global = pl.concat([existing, df_vi_global], how="diagonal")
            df_vi_global = df_vi_global.unique(subset=["feature", "model", "country", "pfs"], keep="last")

        df_vi_global.write_parquet(vi_global_path)

        return df_global, df_vi_global

def add_comb_model(excntry, pfs, data_path, adj):
    
    suffix = get_suffix(adj)
    base = data_path/"ml_model_output"
    results_monthly_path = base / f"{excntry}_{pfs}_pred_month{suffix}.parquet"
    vi_month_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"

    monthly_df = pl.read_parquet(results_monthly_path).filter(pl.col("model") != "COMB")

    if adj == 2:

        comb_df = (
            monthly_df
            .group_by(["row_id", "country", "pfs"])
            .agg([
                pl.col("probability").mean().alias("probability"),
                pl.col("actual").first().alias("actual"),
                pl.col("ret_raw").first().alias("ret_raw"),
                pl.col("ret_cross").first().alias("ret_cross"),
            ])
            .with_columns([
                pl.when(pl.col("probability") > 0.5)
                .then(1)
                .otherwise(-1)
                .cast(pl.Float64)
                .alias("prediction"),
                pl.lit("COMB").alias("model")
            ])
        )

    else:
        comb_df = (monthly_df
                .group_by(["row_id", "country", "pfs"])
                .agg([
                    pl.col("prediction").mean().alias("prediction"),
                    pl.col("actual").first().alias("actual"),
                    pl.col("ret_raw").first().alias("ret_raw")
                    ])
                .with_columns([
                    pl.lit(np.nan).cast(pl.Float64).alias("probability"),
                    pl.lit("COMB").alias("model")]))

    comb_df = comb_df.select(monthly_df.columns)
    monthly_df = pl.concat([monthly_df, comb_df])
    monthly_df.write_parquet(results_monthly_path)

    results_global_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_eval_global{suffix}.parquet"
    global_df = pl.read_parquet(results_global_path).filter(pl.col("model") != "COMB")

    out_corr = eval_correlation(
            df=comb_df.to_pandas(),
            excntry=excntry,
            pfs=pfs,
            adj=adj,
            data_path=data_path
        )

    y_true = comb_df["actual"].to_numpy()
    y_pred = comb_df["prediction"].to_numpy()

    out_score = (
        eval_classification(y_true.astype(int), y_pred.astype(int)) if adj == 2
        else eval_regression(y_true, y_pred))

    comb_global = pl.DataFrame({
        "country": [excntry],
        "pfs": [pfs],
        "model": ["COMB"],
        "r2": [out_score["r2"]],
        "balanced_accuracy": [out_score["balanced_accuracy"]],
        "spearman": [out_corr["spearman"]],
        "pearson": [out_corr["pearson"]]})

    global_df = pl.concat([global_df, comb_global])
    global_df.write_parquet(results_global_path)


    vi_month_df = pl.read_parquet(vi_month_path).filter(pl.col("model") != "COMB")

    comb_vi_month_df = (
        vi_month_df
        .group_by(["feature", "country", "pfs", "test_start"])
        .agg(pl.col("importance").mean().alias("importance"))
        .with_columns(pl.lit("COMB").alias("model"))
    )

    comb_vi_month_df = comb_vi_month_df.select(vi_month_df.columns)

    vi_month_df = pl.concat([vi_month_df, comb_vi_month_df], how="diagonal")
    vi_month_df.write_parquet(vi_month_path)

    vi_global_path = base / f"{excntry}_{pfs}_vi_global{suffix}.parquet"

    vi_global_df = (
        vi_month_df
        .group_by(["feature", "model", "country", "pfs"])
        .agg(pl.col("importance").mean().alias("importance"))
        .with_columns(
            (
                pl.col("importance")
                / pl.col("importance").sum().over(["model", "country", "pfs"])
            )
            .fill_nan(0)
            .fill_null(0)
            .alias("importance")
        )
    )

    vi_global_df.write_parquet(vi_global_path)

    return monthly_df, global_df, vi_month_df, vi_global_df

def sent_port_ret(all_port_ret_monthly, sent, base, excntry, pfs, n_buckets, suffix):

    sent = (
        sent
        .select(["yearmo", "SENT"])
        .with_columns(
            pl.col("yearmo").cast(pl.Int64).alias("yearmo")
        )
        .with_columns(
            pl.date(
                (pl.col("yearmo") // 100).cast(pl.Int32),
                (pl.col("yearmo") % 100).cast(pl.Int32),
                1
            )
            .dt.month_end()
            .alias("eom")
        )
        .select(["eom", "SENT"])
        .drop_nulls()
        .unique("eom")
    )

    sent_median = (
        all_port_ret_monthly
        .select("eom")
        .unique()
        .join(sent, on="eom", how="inner")
        .select(pl.col("SENT").median())
        .item()
    )

    all_sent_monthly_df = (
        all_port_ret_monthly
        .join(sent, on="eom", how="inner")
        .with_columns([
            pl.when(pl.col("SENT") > sent_median)
            .then(pl.lit("Bull"))
            .otherwise(pl.lit("Bear"))
            .alias("period"),

            pl.lit(sent_median).alias("median"),
            pl.col("SENT").alias("sentiment"),
        ])
    )


    all_sent_global_df = (
        all_sent_monthly_df
        .group_by(["model", "period", "bucket"])
        .agg(
            pl.col("mean_ret_bucket_monthly")
            .mean()
            .alias("mean_ret_bucket_period")
        )
        .with_columns([
            pl.lit(excntry).alias("excntry"),
            pl.lit(pfs).alias("pfs"),
            pl.lit(n_buckets).alias("n_buckets"),
            pl.lit(sent_median).alias("median"),
        ])
        .with_columns(
            (
                pl.col("mean_ret_bucket_period")
                .filter(pl.col("bucket") == n_buckets)
                .first()
                .over(["model", "period"])
                -
                pl.col("mean_ret_bucket_period")
                .filter(pl.col("bucket") == 1)
                .first()
                .over(["model", "period"])
            ).alias("hml_ret_period")
        )
        .sort(["model", "period", "bucket"])
    )

    out_path_monthly = (
        base / f"{excntry}_{pfs}_{n_buckets}_port_ret_sent_month{suffix}.parquet"
    )

    out_path_global = (
        base / f"{excntry}_{pfs}_{n_buckets}_port_ret_sent_global{suffix}.parquet"
    )

    all_sent_monthly_df.write_parquet(out_path_monthly)
    all_sent_global_df.write_parquet(out_path_global)

    return all_sent_monthly_df, all_sent_global_df

def build_strategy_returns(data_path, excntry, pfs, n_buckets, adj):
    
    suffix = get_suffix(adj)

    base = data_path / "portfolio_returns"
        
    meta_path = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet"
    model_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_pred_month{suffix}.parquet"

    monthly_lst = []
    global_lst = []
    bucket_lst = []

    ml_out = pl.read_parquet(model_path)
    meta = pl.read_parquet(meta_path)

    eom_pred = ml_out.join(meta["row_id", "eom"], on="row_id", how="left")

    models = eom_pred.select("model").unique().to_series().to_list()

    def run_sort(df, sort_col, label):

        df = (
            df
            .with_columns(
                (
                    pl.col(sort_col)
                    .rank(method="average")
                    .over("eom") / pl.len().over("eom")
                ).alias("rank_pct")
            )
            .with_columns(
                (
                    (pl.col("rank_pct") * n_buckets)
                    .ceil()
                    .clip(1, n_buckets)
                    .cast(pl.Int64)
                ).alias("bucket")
            )
            .with_columns(pl.lit(label).alias("model"))
        )

        port_ret_monthly = (
            df
            .group_by("eom", "bucket")
            .agg(pl.col("ret_raw").mean().alias("mean_ret_bucket_monthly"))
            .with_columns(pl.lit(excntry).alias("excntry"))
            .with_columns(pl.lit(pfs).alias("pfs"))
            .with_columns(pl.lit(n_buckets).alias("n_buckets"))
            .with_columns(pl.lit(label).alias("model"))
        )

        port_ret_monthly = port_ret_monthly.with_columns(
            (
                pl.col("mean_ret_bucket_monthly")
                .filter(pl.col("bucket") == n_buckets)
                .first().over("eom")
                -
                pl.col("mean_ret_bucket_monthly")
                .filter(pl.col("bucket") == 1)
                .first().over("eom")
            ).alias("hml_ret_monthly")
        )

        port_ret_global = (
            port_ret_monthly
            .group_by("bucket")
            .agg(
                pl.col("mean_ret_bucket_monthly").mean()
                .alias("mean_ret_bucket_global")
            )
            .with_columns(pl.lit(excntry).alias("excntry"))
            .with_columns(pl.lit(pfs).alias("pfs"))
            .with_columns(pl.lit(n_buckets).alias("n_buckets"))
            .with_columns(pl.lit(label).alias("model"))
        )

        port_ret_global = port_ret_global.with_columns(
            (
                pl.col("mean_ret_bucket_global")
                .filter(pl.col("bucket") == n_buckets)
                .first()
                -
                pl.col("mean_ret_bucket_global")
                .filter(pl.col("bucket") == 1)
                .first()
            ).alias("hml_ret_global")
        )

        return df, port_ret_monthly, port_ret_global

    oracle_df = (eom_pred
                 .select(["row_id", "eom", "ret_raw", "country", "pfs"])
                 .unique())

    oracle_buckets, oracle_monthly, oracle_global = run_sort(oracle_df, "ret_raw", "ORACLE")

    bucket_lst.append(oracle_buckets)
    monthly_lst.append(oracle_monthly)
    global_lst.append(oracle_global)


    rng = np.random.default_rng(42)

    random_df = (eom_pred.select(["row_id", "eom", "ret_raw", "country", "pfs"])
                 .unique()
                 .sort(["eom", "row_id"]))

    random_df = random_df.with_columns(pl.Series("random_sort", rng.random(random_df.height)))

    random_buckets, random_monthly, random_global = run_sort(random_df, "random_sort", "RANDOM")

    bucket_lst.append(random_buckets)
    monthly_lst.append(random_monthly)
    global_lst.append(random_global)

    for model in models:

        df_model = eom_pred.filter(pl.col("model") == model)

        sort_col = "probability" if adj == 2 else "prediction"

        ml_buckets, port_ret_monthly, port_ret_global = run_sort(df_model, sort_col, model)

        bucket_lst.append(ml_buckets)
        monthly_lst.append(port_ret_monthly)
        global_lst.append(port_ret_global)

    all_port_ret_bucket = pl.concat(bucket_lst, how="diagonal_relaxed")
    all_port_ret_monthly = pl.concat(monthly_lst)
    all_port_ret_global = pl.concat(global_lst)

    sent = (
        pl.read_excel(data_path / "sentiment.xlsx", sheet_name="DATA")
        .select(["yearmo", "SENT"])
    )

    all_port_ret_sent_monthly, all_port_ret_sent_global = sent_port_ret(
        all_port_ret_monthly=all_port_ret_monthly,
        sent=sent,
        base=base,
        excntry=excntry,
        pfs=pfs,
        n_buckets=n_buckets,
        suffix=suffix
    )

    out_path_mon = base/f"{excntry}_{pfs}_{n_buckets}_port_ret_month{suffix}.parquet"
    out_path_buck = base/f"{excntry}_{pfs}_{n_buckets}_factor_to_bucket{suffix}.parquet"
    out_path_gl = base/f"{excntry}_{pfs}_{n_buckets}_port_ret_global{suffix}.parquet"

    all_port_ret_monthly.write_parquet(out_path_mon)
    all_port_ret_bucket.write_parquet(out_path_buck)
    all_port_ret_global.write_parquet(out_path_gl)

    return all_port_ret_global, all_port_ret_monthly, all_port_ret_bucket, all_port_ret_sent_monthly, all_port_ret_sent_global


def compute_turnover(df):

    df_sets = (
        df.group_by("eom")
        .agg(pl.col("characteristics").alias("members"))
        .sort("eom")
    )

    df_sets = df_sets.with_columns(
        pl.col("members").shift(1).alias("prev_members")
    )

    df_sets = df_sets.with_columns(
        pl.struct(["members", "prev_members"]).map_elements(
            lambda x: (
                len(set(x["members"]).intersection(set(x["prev_members"])))
                if x["prev_members"] is not None else None
            )
        ).alias("overlap")
    )

    df_sets = df_sets.with_columns(
        (
            1 - pl.col("overlap") / pl.col("members").list.len()
        ).alias("turnover")
    )

    return df_sets.drop_nulls()

def eval_strategy_returns(
        data_path, excntry, pfs, n_buckets, adj):

        suffix = get_suffix(adj)
        
        base = data_path / "portfolio_returns"

        bucket_path = base/f"{excntry}_{pfs}_{n_buckets}_factor_to_bucket{suffix}.parquet"
        return_path = base/f"{excntry}_{pfs}_{n_buckets}_port_ret_month{suffix}.parquet"
        meta_path = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet"


        meta_data = pl.read_parquet(meta_path).select(["row_id", "characteristics"])
        factor_to_bucket = pl.read_parquet(bucket_path)
        strategy_returns = pl.read_parquet(return_path)

        ew_strategy_returns = (
                    factor_to_bucket
                    .select(["eom", "row_id", "ret_raw"]).unique()
                    .group_by("eom")
                    .agg(pl.col("ret_raw").mean().alias("ew_strategy_ret")))

        random_hml_returns = (
            strategy_returns
            .filter(pl.col("model") == "RANDOM")
            .group_by("eom")
            .agg(
                pl.col("hml_ret_monthly")
                .first()
                .alias("random_hml_ret"))
        )

        random_bucket_returns = (
            strategy_returns
            .filter(pl.col("model") == "RANDOM")
            .select(["eom", "bucket", "mean_ret_bucket_monthly"])
            .rename({"mean_ret_bucket_monthly": "random_bucket_ret"})
        )
        
        ff_monthly = pl.read_parquet(data_path/"other_input"/"ff_monthly.parquet"
                                             ).with_columns(pl.col("eom").cast(pl.Date))
        
        models = (strategy_returns.filter(pl.col("model") != "RANDOM")
                  .select("model").unique().to_series().to_list())

        regression_lst = []
        regression_bucket_lst = []
        turnover_lst = []
        
        for model in models:
            
            strat_model = strategy_returns.filter(pl.col("model") == model)

            df_regress = (
                strat_model
                .group_by("eom")
                .agg(pl.col("hml_ret_monthly").first().alias("strategy_ret"))
                .join(ew_strategy_returns, on="eom", how="left")
                .join(random_hml_returns, on="eom", how="left")
                .join(ff_monthly, on="eom", how="left")
                .sort("eom")
                .to_pandas()
                )

            df_regress = df_regress.dropna(
                subset=["strategy_ret",
                        "ew_strategy_ret",
                        "random_hml_ret",
                        "mkt", "smb", "hml", "rmw", "cma", "wml"])

            X_ff = df_regress[["mkt", "smb", "hml", "rmw", "cma", "wml"]]
            X_ff = sm.add_constant(X_ff)
            
            X_ew = df_regress[["ew_strategy_ret"]]
            X_ew = sm.add_constant(X_ew)

            X_random = df_regress[["random_hml_ret"]]
            X_random = sm.add_constant(X_random)

            y = df_regress["strategy_ret"]
            X_mean = pd.DataFrame({"const": 1}, index=y.index)

            regress_mean = sm.OLS(y, X_mean).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regress_ff = sm.OLS(y, X_ff).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            
            regress_ew = sm.OLS(y, X_ew).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regress_random = sm.OLS(y, X_random).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            
            regression_lst.append({
                "country": excntry,
                "pfs": pfs,
                "model": model,
                "n_buckets": n_buckets,
                "alpha_mean": regress_mean.params["const"],
                "tstat_mean": regress_mean.tvalues["const"],
                "alpha_ff": regress_ff.params["const"],
                "tstat_ff": regress_ff.tvalues["const"],
                "alpha_ew": regress_ew.params["const"],
                "tstat_ew": regress_ew.tvalues["const"],
                "alpha_random": regress_random.params["const"],
                "tstat_random": regress_random.tvalues["const"]})

            bucket_model = factor_to_bucket.filter(pl.col("model") == model)
            
            df_turnover = (
                bucket_model
                .select(["eom", "row_id", "bucket"])
                .join(meta_data, on="row_id", how="left"))
            
            bottom = (df_turnover.filter(pl.col("bucket")==1))
            top = (df_turnover.filter(pl.col("bucket")==n_buckets))

            avg_top = compute_turnover(top)["turnover"].mean()
            avg_bot = compute_turnover(bottom)["turnover"].mean()

            turnover_lst.append({
                "country": excntry,
                "pfs": pfs,
                "model": model,
                "n_buckets": n_buckets,
                "bottom_dec": avg_bot,
                "top_dec": avg_top,})
            
            df_b = (
                strat_model
                .select(["eom", "bucket", "mean_ret_bucket_monthly"])
                .rename({"mean_ret_bucket_monthly": "ret"})
                .join(ew_strategy_returns, on="eom", how="left")
                .join(random_bucket_returns, on=["eom", "bucket"], how="left")
                .join(ff_monthly, on="eom", how="left")
                .sort("eom")
                .to_pandas()
                )

            for b in range(1, n_buckets + 1):

                df_regress_bucket = df_b[df_b["bucket"] == b].dropna(
                    subset=[
                        "ret",
                        "ew_strategy_ret",
                        "random_bucket_ret",
                        "mkt", "smb", "hml", "rmw", "cma", "wml"])

                y = df_regress_bucket["ret"]
                
                X_ff = df_regress_bucket[["mkt", "smb", "hml", "rmw", "cma", "wml"]]
                X_ff = sm.add_constant(X_ff)

                X_ew = df_regress_bucket[["ew_strategy_ret"]]
                X_ew = sm.add_constant(X_ew)

                X_random = df_regress_bucket[["random_bucket_ret"]]
                X_random = sm.add_constant(X_random)

                regress_ff = sm.OLS(y, X_ff).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                regress_ew = sm.OLS(y, X_ew).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                regress_random = sm.OLS(y, X_random).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                
                regression_bucket_lst.append({
                        "country": excntry,
                        "pfs": pfs,
                        "model": model,
                        "bucket": b,
                        "n_buckets": n_buckets,
                        "alpha_ff": regress_ff.params["const"],
                        "tstat_ff": regress_ff.tvalues["const"],
                        "alpha_ew": regress_ew.params["const"],
                        "tstat_ew": regress_ew.tvalues["const"],
                        "alpha_random": regress_random.params["const"],
                        "tstat_random": regress_random.tvalues["const"]
                    })
                
        regressions_strategy_path = base / f"{excntry}_{pfs}_{n_buckets}_regress_strat{suffix}.parquet"
        regressions_bucket_path = base / f"{excntry}_{pfs}_{n_buckets}_regress_buck{suffix}.parquet"
        turnover_path = base / f"{excntry}_{pfs}_{n_buckets}_turnover{suffix}.parquet"

        all_regressions_strategy = pl.DataFrame(regression_lst)
        all_regressions_strategy.write_parquet(regressions_strategy_path)
        
        all_turnovers = pl.DataFrame(turnover_lst)
        all_turnovers.write_parquet(turnover_path)
        
        all_regressions_bucket = pl.DataFrame(regression_bucket_lst)
        all_regressions_bucket.write_parquet(regressions_bucket_path)

        return all_regressions_strategy, all_turnovers, all_regressions_bucket


def eval_strategy_returns_sent(
        data_path, excntry, pfs, n_buckets, adj):

    suffix = get_suffix(adj)

    base = data_path / "portfolio_returns"

    bucket_path = base / f"{excntry}_{pfs}_{n_buckets}_factor_to_bucket{suffix}.parquet"
    period_return_path = base / f"{excntry}_{pfs}_{n_buckets}_port_ret_sent_month{suffix}.parquet"

    factor_to_bucket = pl.read_parquet(bucket_path)
    strategy_returns = pl.read_parquet(period_return_path)

    ew_strategy_returns = (
        factor_to_bucket
        .select(["eom", "row_id", "ret_raw"]).unique()
        .group_by("eom")
        .agg(pl.col("ret_raw").mean().alias("ew_strategy_ret"))
    )

    random_hml_returns = (
        strategy_returns
        .filter(pl.col("model") == "RANDOM")
        .group_by(["period", "eom"])
        .agg(
            pl.col("hml_ret_monthly").first().alias("random_hml_ret")
        )
    )

    ff_monthly = (
        pl.read_parquet(data_path / "other_input" / "ff_monthly.parquet")
        .with_columns(pl.col("eom").cast(pl.Date))
    )

    models = (
        strategy_returns.filter(pl.col("model") != "RANDOM")
        .select("model").unique().to_series().to_list())

    periods = (
        strategy_returns.select("period").unique().to_series().to_list())

    regression_lst = []

    for period in periods:

        strategy_period = strategy_returns.filter(pl.col("period") == period)

        for model in models:

            strat_model = strategy_period.filter(pl.col("model") == model)

            if strat_model.height == 0:
                continue

            df_regress = (
                strat_model
                .group_by(["period", "eom"])
                .agg(pl.col("hml_ret_monthly").first().alias("strategy_ret"))
                .join(ew_strategy_returns, on="eom", how="left")
                .join(random_hml_returns, on=["period", "eom"], how="left")
                .join(ff_monthly, on="eom", how="left")
                .sort("eom")
                .to_pandas()
            )

            df_regress = df_regress.dropna(
                subset=["strategy_ret", "ew_strategy_ret","random_hml_ret",
                        "mkt", "smb", "hml", "rmw", "cma", "wml"]
            )

            if len(df_regress) == 0:
                continue

            y = df_regress["strategy_ret"]

            X_mean = pd.DataFrame({"const": 1}, index=y.index)

            X_ff = df_regress[["mkt", "smb", "hml", "rmw", "cma", "wml"]]
            X_ff = sm.add_constant(X_ff)

            X_ew = df_regress[["ew_strategy_ret"]]
            X_ew = sm.add_constant(X_ew)

            X_random = df_regress[["random_hml_ret"]]
            X_random = sm.add_constant(X_random)

            regress_mean = sm.OLS(y, X_mean).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regress_ff = sm.OLS(y, X_ff).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regress_ew = sm.OLS(y, X_ew).fit(cov_type="HAC",cov_kwds={"maxlags": 6})

            regress_random = sm.OLS(y, X_random).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regression_lst.append({
                "country": excntry,
                "pfs": pfs,
                "model": model,
                "period": period,
                "n_buckets": n_buckets,
                "n_months": len(df_regress),
                "alpha_mean": regress_mean.params["const"],
                "tstat_mean": regress_mean.tvalues["const"],
                "alpha_ff": regress_ff.params["const"],
                "tstat_ff": regress_ff.tvalues["const"],
                "alpha_ew": regress_ew.params["const"],
                "tstat_ew": regress_ew.tvalues["const"],
                "alpha_random": regress_random.params["const"],
                "tstat_random": regress_random.tvalues["const"],
            })

    all_period_regressions = pl.DataFrame(regression_lst)

    out_path = base/f"{excntry}_{pfs}_{n_buckets}_regress_strat_sent{suffix}.parquet"

    all_period_regressions.write_parquet(out_path)

    return all_period_regressions

def compute_backtest_metrics(data_path, excntry, pfs, n_buckets, adj):
    
    suffix = get_suffix(adj)
    base = data_path / "portfolio_returns"

    bucket_path = base / f"{excntry}_{pfs}_{n_buckets}_factor_to_bucket{suffix}.parquet"
    return_path = base / f"{excntry}_{pfs}_{n_buckets}_port_ret_month{suffix}.parquet"
    meta_path = data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet"

    factor_to_bucket = pl.read_parquet(bucket_path).filter(pl.col("model") != "ORACLE")
    strategy_returns = pl.read_parquet(return_path).filter(pl.col("model") != "ORACLE")

    meta = (
        pl.read_parquet(meta_path)
        .select(["row_id", "characteristics"])
    )
    
    factor_to_bucket = factor_to_bucket.join(meta, on="row_id", how="left")

    def net_performance(ret):
        ret = pd.Series(ret).dropna()

        if len(ret) == 0:
            return np.nan

        return (1 + ret).prod() - 1
    
    def sharpe_ratio(ret):
        ret = pd.Series(ret).dropna()

        if len(ret) < 2:
            return np.nan

        std = ret.std(ddof=1)

        if std == 0:
            return np.nan

        return ret.mean() / std * np.sqrt(12)
    
    def sortino_ratio(ret):
        ret = pd.Series(ret).dropna()
        downside = ret[ret < 0]

        if len(ret) < 2 or len(downside) < 2:
            return np.nan

        downside_std = downside.std(ddof=1)

        if downside_std == 0:
            return np.nan

        return ret.mean() / downside_std * np.sqrt(12)

    def max_drawdown(ret):
        ret = pd.Series(ret).dropna()

        if len(ret) == 0:
            return np.nan

        wealth = (1 + ret).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1

        return drawdown.min()

    def streak_stats(condition):
        condition = pd.Series(condition).dropna().astype(bool)

        streaks = []
        current = 0

        for val in condition:
            if val:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0

        if current > 0:
            streaks.append(current)

        if len(streaks) == 0:
            return 0.0, 0

        return float(np.mean(streaks)), int(np.max(streaks))
    
    active = (
        factor_to_bucket
        .filter(pl.col("bucket").is_in([1, n_buckets]))
        .with_columns(
            pl.when(pl.col("bucket") == n_buckets)
            .then(pl.lit("LONG"))
            .otherwise(pl.lit("SHORT"))
            .alias("side")
        )
        .select(["model", "eom", "characteristics", "side"])
        .unique()
        .sort(["model", "eom", "side", "characteristics"])
    )

    monthly_positions = (
        active
        .group_by(["model", "eom"])
        .agg([
            pl.len().alias("positions"),
            (pl.col("side") == "LONG").sum().alias("long_positions"),
            (pl.col("side") == "SHORT").sum().alias("short_positions"),
            pl.struct(["characteristics", "side"]).alias("position_set"),
        ])
        .sort(["model", "eom"])
    )

    monthly_positions = monthly_positions.with_columns(
        pl.col("position_set").shift(1).over("model").alias("prev_position_set")
    )

    monthly_positions = monthly_positions.with_columns([
        pl.struct(["position_set", "prev_position_set"]).map_elements(
            lambda x: (
                len(
                    set((p["characteristics"], p["side"]) for p in x["position_set"])
                    -
                    set((p["characteristics"], p["side"]) for p in x["prev_position_set"])
                )
                if x["prev_position_set"] is not None else None
            ),
            return_dtype=pl.Int64
        ).alias("entry_trades"),

        pl.struct(["position_set", "prev_position_set"]).map_elements(
            lambda x: (
                len(
                    set((p["characteristics"], p["side"]) for p in x["prev_position_set"])
                    -
                    set((p["characteristics"], p["side"]) for p in x["position_set"])
                )
                if x["prev_position_set"] is not None else None
            ),
            return_dtype=pl.Int64
        ).alias("exit_trades"),
    ])

    monthly_positions = monthly_positions.with_columns(
        (
            pl.col("entry_trades") + pl.col("exit_trades")
        ).alias("trades")
    )

    monthly_positions = monthly_positions.with_columns([
        (
            pl.col("entry_trades") + pl.col("exit_trades")
        ).alias("trades"),

        (
            (pl.col("entry_trades") + pl.col("exit_trades"))
            / pl.col("positions")
        ).alias("turnover")
    ])

    monthly_positions = monthly_positions.drop(["position_set", "prev_position_set"])

    monthly_returns = (
        strategy_returns
        .group_by(["model", "eom"])
        .agg(pl.col("hml_ret_monthly").first().alias("strategy_ret"))
        .join(monthly_positions, on=["model", "eom"], how="left")
        .sort(["model", "eom"])
    )

    monthly_pd = monthly_returns.to_pandas()
    monthly_pd["eom"] = pd.to_datetime(monthly_pd["eom"])

    metrics_lst = []

    for model, df_model in monthly_pd.groupby("model"):

        df_model = df_model.sort_values("eom").copy()
        ret = df_model["strategy_ret"].dropna()

        if len(ret) == 0:
            continue

        wins = ret[ret > 0]
        losses = ret[ret < 0]

        win_streak_avg, win_streak_max = streak_stats(ret > 0)
        loss_streak_avg, loss_streak_max = streak_stats(ret < 0)

        average_win = wins.mean() if len(wins) > 0 else np.nan
        average_loss = losses.mean() if len(losses) > 0 else np.nan

        reward_to_risk_ratio = (
            abs(average_win / average_loss)
            if pd.notna(average_win)
            and pd.notna(average_loss)
            and average_loss != 0
            else np.nan
        )

        metrics_lst.append({
            "country": excntry,
            "pfs": pfs,
            "model": model,
            "n_buckets": n_buckets,
            "average_return": ret.mean(),
            "return_std": ret.std(ddof=1) if len(ret) > 1 else np.nan,
            "sharpe_ratio": sharpe_ratio(ret),
            "sortino_ratio": sortino_ratio(ret),
            #"net_performance": net_performance(ret),
            "max_drawdown": max_drawdown(ret),
            "wins": len(wins),
            "losses": len(losses),
            "average_win": average_win,
            "average_loss": average_loss,
            "reward_to_risk_ratio": reward_to_risk_ratio,
            #"loss_std": losses.std(ddof=1) if len(losses) > 1 else np.nan,
            #"win_streak_avg": win_streak_avg,
            #"win_streak_max": win_streak_max,
            #"loss_streak_avg": loss_streak_avg,
            #"loss_streak_max": loss_streak_max,
            "trades_per_month": df_model["trades"].mean(),
            #"active_positions": df_model["positions"].mean(),
            "turnover": df_model["turnover"].mean(),
            #"entry_trades": df_model["entry_trades"].mean(),
            #"exit_trades": df_model["exit_trades"].mean(),
            #"long_positions": df_model["long_positions"].mean(),
            #"short_positions": df_model["short_positions"].mean(),
            #"n_months": len(ret),
        })

    metrics_global = pl.from_pandas(pd.DataFrame(metrics_lst))

    out_path_month = base / f"{excntry}_{pfs}_{n_buckets}_backtest_metrics_month{suffix}.parquet"
    out_path_global = base / f"{excntry}_{pfs}_{n_buckets}_backtest_metrics_global{suffix}.parquet"

    monthly_returns.write_parquet(out_path_month)
    metrics_global.write_parquet(out_path_global)

    return monthly_returns, metrics_global


def compute_backtest_metrics_sent(data_path, excntry, pfs, n_buckets, adj):

    suffix = get_suffix(adj)
    base = data_path / "portfolio_returns"

    sent_return_path = (
        base / f"{excntry}_{pfs}_{n_buckets}_port_ret_sent_month{suffix}.parquet"
    )

    bucket_path = (
        base / f"{excntry}_{pfs}_{n_buckets}_factor_to_bucket{suffix}.parquet"
    )

    meta_path = (
        data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet"
    )

    strategy_returns = (
        pl.read_parquet(sent_return_path)
        .filter(pl.col("model") != "ORACLE")
    )

    factor_to_bucket = (
        pl.read_parquet(bucket_path)
        .filter(pl.col("model") != "ORACLE")
    )

    meta = (
        pl.read_parquet(meta_path)
        .select(["row_id", "characteristics"])
    )

    factor_to_bucket = factor_to_bucket.join(meta, on="row_id", how="left")

    def sharpe_ratio(ret):
        ret = pd.Series(ret).dropna()
        if len(ret) < 2:
            return np.nan
        std = ret.std(ddof=1)
        if std == 0:
            return np.nan
        return ret.mean() / std * np.sqrt(12)

    def sortino_ratio(ret):
        ret = pd.Series(ret).dropna()
        downside = ret[ret < 0]
        if len(ret) < 2 or len(downside) < 2:
            return np.nan
        downside_std = downside.std(ddof=1)
        if downside_std == 0:
            return np.nan
        return ret.mean() / downside_std * np.sqrt(12)

    def max_drawdown(ret):
        ret = pd.Series(ret).dropna()
        if len(ret) == 0:
            return np.nan
        wealth = (1 + ret).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1
        return drawdown.min()

    active = (
        factor_to_bucket
        .filter(pl.col("bucket").is_in([1, n_buckets]))
        .with_columns(
            pl.when(pl.col("bucket") == n_buckets)
            .then(pl.lit("LONG"))
            .otherwise(pl.lit("SHORT"))
            .alias("side")
        )
        .select(["model", "eom", "characteristics", "side"])
        .unique()
        .sort(["model", "eom", "side", "characteristics"])
    )

    monthly_positions = (
        active
        .group_by(["model", "eom"])
        .agg([
            pl.len().alias("positions"),
            (pl.col("side") == "LONG").sum().alias("long_positions"),
            (pl.col("side") == "SHORT").sum().alias("short_positions"),
            pl.struct(["characteristics", "side"]).alias("position_set"),
        ])
        .sort(["model", "eom"])
        .with_columns(
            pl.col("position_set")
            .shift(1)
            .over("model")
            .alias("prev_position_set")
        )
        .with_columns([
            pl.struct(["position_set", "prev_position_set"]).map_elements(
                lambda x: (
                    len(
                        set((p["characteristics"], p["side"]) for p in x["position_set"])
                        -
                        set((p["characteristics"], p["side"]) for p in x["prev_position_set"])
                    )
                    if x["prev_position_set"] is not None else None
                ),
                return_dtype=pl.Int64
            ).alias("entry_trades"),

            pl.struct(["position_set", "prev_position_set"]).map_elements(
                lambda x: (
                    len(
                        set((p["characteristics"], p["side"]) for p in x["prev_position_set"])
                        -
                        set((p["characteristics"], p["side"]) for p in x["position_set"])
                    )
                    if x["prev_position_set"] is not None else None
                ),
                return_dtype=pl.Int64
            ).alias("exit_trades"),
        ])
        .with_columns([
            (pl.col("entry_trades") + pl.col("exit_trades")).alias("trades"),
            (
                (pl.col("entry_trades") + pl.col("exit_trades"))
                / pl.col("positions")
            ).alias("turnover"),
        ])
        .drop(["position_set", "prev_position_set"])
    )

    monthly_returns = (
        strategy_returns
        .group_by(["model", "period", "eom"])
        .agg([
            pl.col("hml_ret_monthly").first().alias("strategy_ret"),
            pl.col("sentiment").first().alias("sentiment"),
            pl.col("median").first().alias("median"),
        ])
        .join(monthly_positions, on=["model", "eom"], how="left")
        .sort(["model", "period", "eom"])
    )

    monthly_pd = monthly_returns.to_pandas()
    monthly_pd["eom"] = pd.to_datetime(monthly_pd["eom"])

    metrics_lst = []

    for (model, period), df_group in monthly_pd.groupby(["model", "period"]):

        df_group = df_group.sort_values("eom").copy()
        ret = df_group["strategy_ret"].dropna()

        if len(ret) == 0:
            continue

        wins = ret[ret > 0]
        losses = ret[ret < 0]

        average_win = wins.mean() if len(wins) > 0 else np.nan
        average_loss = losses.mean() if len(losses) > 0 else np.nan

        reward_to_risk_ratio = (
            abs(average_win / average_loss)
            if pd.notna(average_win)
            and pd.notna(average_loss)
            and average_loss != 0
            else np.nan
        )

        metrics_lst.append({
            "country": excntry,
            "pfs": pfs,
            "model": model,
            "period": period,
            "n_buckets": n_buckets,
            "n_months": len(ret),
            "average_return": ret.mean(),
            "return_std": ret.std(ddof=1) if len(ret) > 1 else np.nan,
            "sharpe_ratio": sharpe_ratio(ret),
            "sortino_ratio": sortino_ratio(ret),
            "max_drawdown": max_drawdown(ret),
            "wins": len(wins),
            "losses": len(losses),
            "average_win": average_win,
            "average_loss": average_loss,
            "reward_to_risk_ratio": reward_to_risk_ratio,
            "trades_per_month": df_group["trades"].mean(),
            "turnover": df_group["turnover"].mean(),
            "long_positions": df_group["long_positions"].mean(),
            "short_positions": df_group["short_positions"].mean(),
            "avg_sentiment": df_group["sentiment"].mean(),
            "sentiment_median": df_group["median"].iloc[0],
        })

    metrics_sent = pl.from_pandas(pd.DataFrame(metrics_lst))

    out_path_month = (
        base / f"{excntry}_{pfs}_{n_buckets}_backtest_metrics_sent_month{suffix}.parquet"
    )

    out_path_global = (
        base / f"{excntry}_{pfs}_{n_buckets}_backtest_metrics_sent_global{suffix}.parquet"
    )

    monthly_returns.write_parquet(out_path_month)
    metrics_sent.write_parquet(out_path_global)

    return monthly_returns, metrics_sent


def latex_pred_perf(
    dfs,
    adjs,
    panel_titles,
    caption: str = "Predictive Performance for the Cross Section of Factor Returns",
    label: str = "tab:pred-perf-panels"):

    model_order = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "COMB"]

    header_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
    }
    score_labels = {
        "r2": r"$R^2_{\mathrm{oos}}$",
        "balanced_accuracy": r"$BACC_{\mathrm{oos}}$",
        "spearman": r"$\bar{\rho}_s$",
        "pearson": r"$\bar{\rho}_p$",
    }

    notes = (
        r"\textbf{Notes:} This table reports the prediction performance measures for machine "
        r"learning models applied to the cross section of factor returns: the out-of-sample "
        r"$R^2$ coefficients ($R^2_{\mathrm{oos}}$), out-of-sample balanced accuracy "
        r"($BACC_{\mathrm{oos}}$), and average monthly Spearman ($\bar{\rho}_s$) and "
        r"Pearson ($\bar{\rho}_p$) correlation coefficients between predicted and realized "
        r"outcomes. $R^2_{\mathrm{oos}}$ and $BACC_{\mathrm{oos}}$ are expressed in "
        r"percentages, and correlation coefficients are reported as decimals. The considered "
        r"models include OLS or LOGIT (Linear), PLS, LASSO, ENET, random forests (RF), GBRT, XGBoost "
        r"(XGB), and forecast combination (COMB). The table reports results for four target transformations: "
        r"next-month value-weighted excess factor returns (Benchmark), cross-sectionally "
        r"demeaned next-month factor returns (Cross-Reg), a binary outperformer indicator "
        r"based on cross-sectionally demeaned returns (Cross-Class), and each factor's "
        r"percentile rank within the monthly cross-section (Rank-Reg). For the regression "
        r"targets, the correlation coefficients are computed between predicted and realized "
        r"target values. For the classification target, they are computed between predicted "
        r"outperformance probabilities and realized cross-sectionally demeaned returns. "
        r"COMB is the equal-weighted average forecast across available models; for "
        r"Cross-Class, it averages predicted outperformance probabilities. "
        r"The sample comprises 153 factors from Jensen, Kelly, and Pedersen (2023); "
        r"the forecasting sample runs from October 1971 to September 2024, "
        r"and the out-of-sample testing period runs from October 1986 to September 2024."

    )

    def build_panel_rows(df, adj):
        score = "balanced_accuracy" if adj == 2 else "r2"
        score_order = [score, "spearman", "pearson"]

        df = df.clone()

        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )

        score_order_existing = [s for s in score_order if s in df.columns]

        df_small = df.select(["model"] + score_order_existing)

        long_df = df_small.unpivot(
            index="model",
            variable_name="score",
            value_name="value",
        )

        long_df = long_df.with_columns(
            pl.when(pl.col("value").is_null())
            .then(pl.lit("-"))
            .otherwise(
                pl.struct(["value", "score"]).map_elements(
                    lambda s: (
                        f"{float(s['value']) * 100:.3f}"
                        if s["score"] in {"r2", "balanced_accuracy"}
                        else f"{float(s['value']):.3f}"
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .alias("value_fmt")
        )

        wide_df = (
            long_df
            .select(["score", "model", "value_fmt"])
            .pivot(
                index="score",
                columns="model",
                values="value_fmt",
                aggregate_function="first",
            )
        )

        wide_df = (
            wide_df
            .with_columns(
                pl.col("score")
                .replace({s: i for i, s in enumerate(score_order_existing)})
                .cast(pl.Int64)
                .alias("_score_order")
            )
            .sort("_score_order")
            .drop("_score_order")
        )

        for model in model_order:
            if model not in wide_df.columns:
                wide_df = wide_df.with_columns(pl.lit("-").alias(model))

        wide_df = wide_df.select(["score"] + model_order)

        wide_df = wide_df.with_columns(
            pl.col("score").replace(score_labels).alias("score")
        )

        rows = []

        for row in wide_df.iter_rows(named=True):
            score_name = row["score"]
            values = [row.get(model, "-") or "-" for model in model_order]
            rows.append(score_name + " & " + " & ".join(values) + r" \\")

        return rows

    col_spec = "C{2cm} " + " ".join(["Z"] * len(model_order))
    n_cols = 1 + len(model_order)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.2}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join(
            rf"\textbf{{{header_labels[model]}}}" for model in model_order
        ) + r" \\",
        r"\midrule",
    ]

    for i, (df, adj, title) in enumerate(zip(dfs, adjs, panel_titles)):
        if i > 0:
            lines.append(r"\addlinespace[0.4em]")
            lines.append(r"\midrule")

        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{title}}}}} \\"
        )

        lines.extend(build_panel_rows(df, adj))

    lines.extend([
        r"\bottomrule",
        "",
        r"\addlinespace[0.3em]",
        rf"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{",
        r"\scriptsize",
        rf"\textbf{{Notes:}} {notes}}}\\",
        "",
        r"\end{tabularx}",
        "",
        r"\end{table}",
    ])

    return "\n".join(lines)

def latex_strat_perf(
    dfs,
    adjs,
    panel_titles,
    caption: str = "Returns on Machine Learning Factor Portfolios",
    label: str = "tab:bucket-returns-panels",
    scale: float = 100.0,
    decimals: int = 2,
):

    model_order = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "COMB", "ORACLE"]

    header_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
        "ORACLE": "ORACLE",
    }

    notes = (
        r"This table reports monthly returns on decile portfolios of factor "
        r"strategies formed on the predictions of machine learning models. The considered "
        r"models include OLS or LOGIT (Linear), PLS, LASSO, ENET, random forests (RF), GBRT, XGBoost "
        r"(XGB), forecast combination (COMB), and a perfect-foresight model (ORACLE). Low "
        r"(High) indicates the decile of factors with the lowest (highest) predicted value. "
        r"High--Low is the spread portfolio that assumes a long position in the High decile "
        r"and a short position in the Low decile. All returns are expressed in percentages "
        r"per month. The table reports results for four target transformations: next-month "
        r"value-weighted excess factor returns (Benchmark), cross-sectionally demeaned "
        r"next-month factor returns (Cross-Reg), a binary outperformer indicator based on "
        r"cross-sectionally demeaned returns (Cross-Class), and each factor's percentile "
        r"rank within the monthly cross-section (Rank-Reg). COMB is the equal-weighted "
        r"average forecast across available models; for Cross-Class, it averages predicted "
        r"outperformance probabilities. ORACLE sorts factors on realized next-month returns "
        r"and therefore represents an ex-post perfect-foresight benchmark that is not "
        r"implementable in real time. The sample comprises 153 factors from Jensen, Kelly, "
        r"and Pedersen (2023); the forecasting sample runs from October 1971 to September "
        r"2024, and the out-of-sample testing period runs from October 1986 to September "
        r"2024."
)

    def build_panel_rows(df, adj):
        df = df.clone()

        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )

        n_buckets = int(df.select(pl.col("bucket").max()).item())

        bucket_labels = {
            1: "Low",
            n_buckets: "High",
        }

        df_bucket = df.with_columns(
            pl.col("bucket")
            .map_elements(
                lambda x: bucket_labels.get(int(x), str(int(x))),
                return_dtype=pl.Utf8,
            )
            .alias("bucket_label"),
            pl.col("bucket").cast(pl.Int64).alias("_bucket_order"),
            pl.col("mean_ret_bucket_global")
            .map_elements(
                lambda x: f"{float(x) * scale:.{decimals}f}",
                return_dtype=pl.Utf8,
            )
            .alias("value_fmt"),
        )

        wide_df = (
            df_bucket
            .select(["bucket_label", "_bucket_order", "model", "value_fmt"])
            .pivot(
                index=["bucket_label", "_bucket_order"],
                columns="model",
                values="value_fmt",
                aggregate_function="first",
            )
            .sort("_bucket_order")
            .drop("_bucket_order")
            .rename({"bucket_label": "bucket"})
        )

        hml_df = (
            df
            .group_by("model")
            .agg(pl.col("hml_ret_global").first().alias("hml_ret_global"))
            .with_columns(
                pl.lit("High-Low").alias("bucket"),
                pl.col("hml_ret_global")
                .map_elements(
                    lambda x: f"{float(x) * scale:.{decimals}f}",
                    return_dtype=pl.Utf8,
                )
                .alias("value_fmt"),
            )
            .select(["bucket", "model", "value_fmt"])
            .pivot(
                index="bucket",
                columns="model",
                values="value_fmt",
                aggregate_function="first",
            )
        )

        for model in model_order:
            if model not in wide_df.columns:
                wide_df = wide_df.with_columns(pl.lit("-").alias(model))

            if model not in hml_df.columns:
                hml_df = hml_df.with_columns(pl.lit("-").alias(model))

        wide_df = wide_df.select(["bucket"] + model_order)
        hml_df = hml_df.select(["bucket"] + model_order)

        rows = []

        for row in wide_df.iter_rows(named=True):
            bucket = row["bucket"]
            values = [row.get(model, "-") or "-" for model in model_order]
            rows.append(bucket + " & " + " & ".join(values) + r" \\")

        rows.append(r"\midrule")

        for row in hml_df.iter_rows(named=True):
            bucket = row["bucket"]
            values = [row.get(model, "-") or "-" for model in model_order]
            rows.append(bucket + " & " + " & ".join(values) + r" \\")

        return rows

    col_spec = "C{1.5cm} " + " ".join(["Z"] * len(model_order))
    n_cols = 1 + len(model_order)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.2}"
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join(
            rf"\textbf{{{header_labels[model]}}}" for model in model_order
        ) + r" \\",
        r"\midrule",
    ]

    for i, (df, adj, title) in enumerate(zip(dfs, adjs, panel_titles)):
        if i > 0:
            lines.append(r"\addlinespace[0.4em]")

        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{title}}}}} \\"
        )

        lines.extend(build_panel_rows(df, adj))

    lines.extend([
        r"\bottomrule",
        "",
        r"\addlinespace[0.3em]",
        rf"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{",
        r"\scriptsize",
        rf"\textbf{{Notes:}} {notes}}}\\",
        "",
        r"\end{tabularx}",
        "",
        r"\end{table}",
    ])

    return "\n".join(lines)


def latex_strat_alphas(
    dfs,
    adjs,
    panel_titles,
    caption: str = "Statistical Significance of Machine Learning Factor Portfolio Returns",
    label: str = "tab:strat-alphas-panels",
):

    model_order = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "COMB", "ORACLE"]

    header_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
        "ORACLE": "ORACLE",
    }

    row_specs = [
        ("alpha_mean", "tstat_mean", "High-Low"),
        ("alpha_ff", "tstat_ff", r"$\alpha_{F6}$"),
        ("alpha_ew", "tstat_ew", r"$\alpha_{EW}$"),
        ("alpha_random", "tstat_random", r"$\alpha_{\mathrm{R}}$"),
    ]

    notes = (
        r"This table reports monthly returns and alphas on High--Low portfolios of factor "
        r"strategies formed on the predictions of machine learning models. The considered "
        r"models include OLS or LOGIT (Linear), PLS, LASSO, ENET, random forests (RF), "
        r"GBRT, XGBoost (XGB), forecast combination (COMB), and a perfect-foresight model "
        r"(ORACLE). High--Low is the spread portfolio that assumes a long position in the "
        r"decile of factors with the highest predicted value and a short position in the "
        r"decile of factors with the lowest predicted value. "
        r"$\alpha_{F6}$, $\alpha_{EW}$, and $\alpha_{\mathrm{R}}$ are alphas from the "
        r"Fama--French six-factor model, the equal-weighted factor benchmark model, and "
        r"the random-sorting benchmark model, respectively. All returns and alphas are "
        r"expressed in percentages. The numbers in parentheses are Newey--West adjusted "
        r"t-statistics with six lags. The table reports results for four target "
        r"transformations: next-month value-weighted excess factor returns (Benchmark), "
        r"cross-sectionally demeaned next-month factor returns (Cross-Reg), a binary "
        r"outperformer indicator based on cross-sectionally demeaned returns (Cross-Class), "
        r"and each factor's percentile rank within the monthly cross-section (Rank-Reg). "
        r"COMB is the equal-weighted average forecast across available models; for "
        r"Cross-Class, it averages predicted outperformance probabilities. ORACLE sorts "
        r"factors on realized next-month returns and therefore represents an ex-post "
        r"perfect-foresight benchmark that is not implementable in real time. The sample "
        r"comprises 153 factors from Jensen, Kelly, and Pedersen (2023); the forecasting "
        r"sample runs from October 1971 to September 2024, and the out-of-sample testing "
        r"period runs from October 1986 to September 2024."
    )

    def fmt_pct(x):
        if x is None:
            return "-"
        return f"{float(x) * 100:.2f}"

    def fmt_t(x):
        if x is None:
            return "-"
        return f"({float(x):.2f})"

    def build_panel_rows(df, adj):
        df = df.clone()

        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )

        row_map = {row["model"]: row for row in df.iter_rows(named=True)}

        rows = []
        
        for j, (value_col, tstat_col, row_label) in enumerate(row_specs):
            
            value_cells = []
            tstat_cells = []

            for model in model_order:
                row = row_map.get(model, None)

                if row is None:
                    value_cells.append("-")
                    tstat_cells.append("-")
                else:
                    value_cells.append(fmt_pct(row.get(value_col)))
                    tstat_cells.append(fmt_t(row.get(tstat_col)))

            if j > 0:
                rows.append(r"\addlinespace[0.4em]")

            rows.append(row_label + " & " + " & ".join(value_cells) + r" \\")
            rows.append(" & " + " & ".join(tstat_cells) + r" \\")

        return rows

    col_spec = "C{1.8cm} " + " ".join(["Z"] * len(model_order))
    n_cols = 1 + len(model_order)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.0}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        " & " + " & ".join(
            rf"\textbf{{{header_labels[model]}}}" for model in model_order
        ) + r" \\",
        r"\midrule",
    ]

    for i, (df, adj, title) in enumerate(zip(dfs, adjs, panel_titles)):
        if i > 0:
            lines.append(r"\addlinespace[0.4em]")
            lines.append(r"\midrule")

        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{title}}}}} \\"
        )

        lines.extend(build_panel_rows(df, adj))

    lines.extend([
        r"\bottomrule",
        "",
        r"\addlinespace[0.3em]",
        rf"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{",
        r"\scriptsize",
        rf"\textbf{{Notes:}} {notes}}}\\",
        "",
        r"\end{tabularx}",
        "",
        r"\end{table}",
    ])

    return "\n".join(lines)

def latex_strat_metrics(
    dfs,
    adjs,
    panel_titles,
    caption="Performance Metrics for Machine Learning Factor Portfolios",
    label="tab:strat-metrics",
):

    models = [
        "OLS",
        "PLS",
        "LASSO",
        "ENET",
        "RF",
        "GBRT",
        "XGB",
        "COMB",
    ]

    model_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
    }

    metrics_to_show = [
        "average_return",
        "return_std",
        "sharpe_ratio",
        #"sortino_ratio",
        "max_drawdown",
        "wins",
        "losses",
        "average_win",
        "average_loss",
        "reward_to_risk_ratio",
        "trades_per_month",
        "turnover",
    ]

    metric_labels = {
        "average_return": "Return",
        "return_std": "Volatility",
        "sharpe_ratio": "Sharpe ratio",
        #"sortino_ratio": "Sortino ratio",
        "max_drawdown": "Max. drawdown",
        "wins": "Positive months",
        "losses": "Negative months",
        "average_win": "Avg. Win",
        "average_loss": "Avg. Loss",
        "reward_to_risk_ratio": "Reward-to-Risk",
        "trades_per_month": "Trades per month",
        "turnover": "Position turnover",
    }

    percent_metrics = {
        "average_return",
        "return_std",
        "max_drawdown",
        "average_win",
        "average_loss",
        "turnover",
    }

    integer_metrics = {
        "wins",
        "losses",
    }

    def fmt(value, metric):
        if pd.isna(value):
            return "-"

        if metric in percent_metrics:
            return f"{100 * value:.2f}"

        if metric in integer_metrics:
            return f"{int(value)}"

        if metric == "trades_per_month":
            return f"{value:.2f}"

        return f"{value:.3f}"

    def to_pandas_safe(df):
        if hasattr(df, "to_pandas"):
            return df.to_pandas().copy()
        return df.copy()

    n_cols = 1 + len(models)
    tabular_spec = "L{2cm}" + " ".join(["Z"] * len(models))

    notes = (
        r"This table reports backtest performance measures for High--Low portfolios of "
        r"factor strategies formed on the predictions of machine learning models. The "
        r"considered models include OLS or LOGIT (Linear), PLS, LASSO, ENET, random "
        r"forests (RF), GBRT, XGBoost (XGB), and forecast combination (COMB). High--Low "
        r"is the spread portfolio that assumes a long position in the decile of factors "
        r"with the highest predicted value and a short position in the decile of factors "
        r"with the lowest predicted value. Return is the average monthly High--Low return, "
        r"Volatility is the annualized volatility of monthly High--Low returns, and the "
        r"Sharpe ratio is annualized using monthly returns. Max. drawdown is the largest "
        r"peak-to-trough decline in cumulative High--Low portfolio value. Positive months "
        r"and Negative months report the number of months with positive and negative "
        r"High--Low returns. Avg. Win and Avg. Loss are the average positive and negative "
        r"monthly High--Low returns, respectively, and Reward-to-Risk is the ratio of "
        r"Avg. Win to the absolute value of Avg. Loss. Trades per month is the average "
        r"number of factor position entries and exits in the long and short legs. Position "
        r"turnover is the average number of trades divided by the number of active long "
        r"and short positions. Returns, volatility, maximum drawdown, average wins, "
        r"average losses, and position turnover are expressed in percentages. The table "
        r"reports results for four target transformations: next-month value-weighted "
        r"excess factor returns (Benchmark), cross-sectionally demeaned next-month factor "
        r"returns (Cross-Reg), a binary outperformer indicator based on cross-sectionally "
        r"demeaned returns (Cross-Class), and each factor's percentile rank within the "
        r"monthly cross-section (Rank-Reg). COMB is the equal-weighted average forecast "
        r"across available models; for Cross-Class, it averages predicted outperformance "
        r"probabilities. The sample comprises 153 factors from Jensen, Kelly, and "
        r"Pedersen (2023); the forecasting sample runs from October 1971 to September "
        r"2024, and the out-of-sample testing period runs from October 1986 to September "
        r"2024."
    )

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(fr"\caption{{{caption}}}")
    lines.append(fr"\label{{{label}}}")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(fr"\begin{{tabularx}}{{\textwidth}}{{{tabular_spec}}}")
    lines.append(r"\toprule")

    header = (
        " & "
        + " & ".join([fr"\textbf{{{model_labels[m]}}}" for m in models])
        + r" \\"
    )
    lines.append(header)
    lines.append(r"\midrule")

    for df, adj, panel_title in zip(dfs, adjs, panel_titles):
        
        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )
        df_pd = to_pandas_safe(df)

        lines.append(fr"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{panel_title}}}}} \\")

        for metric in metrics_to_show:
            row_vals = []

            for model in models:
                row_df = df_pd.loc[df_pd["model"].eq(model)]

                if row_df.empty:
                    row_vals.append("-")
                else:
                    value = row_df.iloc[0][metric]
                    row_vals.append(fmt(value, metric))

            row = metric_labels[metric]
            row += " & " + " & ".join(row_vals)
            row += r" \\"
            lines.append(row)

        if panel_title != panel_titles[-1]:
            lines.append(r"\addlinespace[0.3em]")
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append("")
    lines.append(r"\addlinespace[0.3em]")
    lines.append(
        fr"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{"
    )
    lines.append(r"\scriptsize")
    lines.append(fr"\textbf{{Notes:}} {notes}}}\\")
    lines.append("")
    lines.append(r"\end{tabularx}")
    lines.append("")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def latex_strat_sent_metrics(
    dfs,
    adjs,
    panel_titles,
    period_order=("Bull", "Bear"),
    panels_per_table: int = 2,
    caption: str = "Backtest Metrics for Machine Learning Factor Portfolios in Bull and Bear Periods",
    label: str = "tab:strat-metrics-bull-bear",
    table_pos: str = "h",
):
    models = [
        "OLS",
        "PLS",
        "LASSO",
        "ENET",
        "RF",
        "GBRT",
        "XGB",
        "COMB",
    ]

    model_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
    }

    metrics_to_show = [
        "average_return",
        "return_std",
        "sharpe_ratio",
        # "sortino_ratio",
        "max_drawdown",
        "wins",
        "losses",
        "average_win",
        "average_loss",
        "reward_to_risk_ratio",
        "trades_per_month",
        "turnover",
    ]

    metric_labels = {
        "average_return": "Return",
        "return_std": "Volatility",
        "sharpe_ratio": "Sharpe ratio",
        # "sortino_ratio": "Sortino ratio",
        "max_drawdown": "Max. drawdown",
        "wins": "Positive months",
        "losses": "Negative months",
        "average_win": "Avg. Win",
        "average_loss": "Avg. Loss",
        "reward_to_risk_ratio": "Reward-to-Risk",
        "trades_per_month": "Trades per month",
        "turnover": "Position turnover",
    }

    percent_metrics = {
        "average_return",
        "return_std",
        "max_drawdown",
        "average_win",
        "average_loss",
        "turnover",
    }

    integer_metrics = {
        "wins",
        "losses",
    }

    notes = (
        r"This table reports backtest performance measures for High--Low factor portfolios "
        r"formed from machine-learning predictions, separately for Bull and Bear periods. "
        r"Bull and Bear months are defined by whether the Baker and Wurgler (2006) "
        r"sentiment index is above or below its median. The considered models are OLS or "
        r"LOGIT (Linear), PLS, LASSO, ENET, RF, GBRT, XGB, and COMB. High--Low is long "
        r"the highest-forecast decile and short the lowest-forecast decile. Return is the "
        r"average monthly return; volatility and Sharpe ratio are annualized from monthly "
        r"returns. Max. drawdown is the largest cumulative peak-to-trough loss. Positive "
        r"and negative months count months with positive and negative returns. Avg. Win "
        r"and Avg. Loss are average positive and negative monthly returns; Reward-to-Risk "
        r"is their absolute ratio. Trades per month counts average entries and exits; "
        r"position turnover scales trades by active long and short positions. Percentage "
        r"metrics are reported in percent. Results are shown for Benchmark, Cross-Reg, "
        r"Cross-Class, and Rank-Reg targets. The sample comprises 153 "
        r"factors from Jensen, Kelly, and Pedersen (2023); the forecasting sample runs "
        r"from October 1971 to September 2024, and the out-of-sample testing period runs "
        r"from October 1986 to September 2024."
    )

    def fmt(value, metric):
        if pd.isna(value):
            return "-"

        if metric in percent_metrics:
            return f"{100 * value:.2f}"

        if metric in integer_metrics:
            return f"{int(value)}"

        if metric == "trades_per_month":
            return f"{value:.2f}"

        return f"{value:.3f}"

    def to_pandas_safe(df):
        if hasattr(df, "to_pandas"):
            return df.to_pandas().copy()
        return df.copy()

    def clean_model_names(df, adj):
        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )
        return df

    def build_period_rows(df_pd, period_label):
        rows = []

        for metric in metrics_to_show:
            row_vals = []

            for model in models:
                row_df = df_pd.loc[
                    df_pd["model"].eq(model) & df_pd["period"].eq(period_label)
                ]

                if row_df.empty:
                    row_vals.append("-")
                else:
                    value = row_df.iloc[0][metric]
                    row_vals.append(fmt(value, metric))

            row = metric_labels[metric]
            row += " & " + " & ".join(row_vals)
            row += r" \\"
            rows.append(row)

        return rows

    def build_panel_rows(df, adj, panel_title):
        df = clean_model_names(df, adj)
        df_pd = to_pandas_safe(df)

        rows = []

        for p, period_label in enumerate(period_order):
            if p > 0:
                rows.append(r"\addlinespace[0.5em]")

            rows.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{panel_title} -- {period_label}}}}} \\"
            )

            rows.extend(build_period_rows(df_pd, period_label))

        return rows

    def add_notes(lines):
        lines.extend([
            r"\bottomrule",
            "",
            r"\addlinespace[0.3em]",
            rf"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{",
            r"\scriptsize",
            rf"\textbf{{Notes:}} {notes}}}\\",
        ])

    def chunks(x, n):
        for start in range(0, len(x), n):
            yield start, x[start:start + n]

    n_cols = 1 + len(models)
    tabular_spec = "L{2cm}" + " ".join(["Z"] * len(models))

    table_blocks = []
    zipped_panels = list(zip(dfs, adjs, panel_titles))

    for chunk_start, panel_chunk in chunks(zipped_panels, panels_per_table):
        table_index = chunk_start // panels_per_table
        is_first_table = table_index == 0

        if is_first_table:
            this_caption = caption
            this_label = label
        else:
            this_caption = rf"{caption} \textit{{(continued)}}"
            this_label = f"{label}-continued-{table_index + 1}"

        lines = [
            rf"\begin{{table}}[{table_pos}]",
            r"\centering",
            rf"\caption{{{this_caption}}}",
            rf"\label{{{this_label}}}",
            r"\scriptsize",
            r"\renewcommand{\arraystretch}{1.0}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{tabular_spec}}}",
            r"\toprule",
            " & " + " & ".join(
                rf"\textbf{{{model_labels[m]}}}" for m in models
            ) + r" \\",
            r"\midrule",
        ]

        for i, (df, adj, title) in enumerate(panel_chunk):
            if i > 0:
                lines.append(r"\addlinespace[0.5em]")
                lines.append(r"\midrule")

            lines.extend(build_panel_rows(df, adj, title))

        add_notes(lines)

        lines.extend([
            r"\end{tabularx}",
            "",
            r"\end{table}",
        ])

        table_blocks.append("\n".join(lines))

        if chunk_start + panels_per_table < len(zipped_panels):
            table_blocks.append(r"\newpage")

    return "\n\n".join(table_blocks)

def plot_cum_perf_buck(
        base_path,
        data_path,
        excntry,
        pfs,
        n_buckets,
        adj,
        save=True,
        show=False
):


    suffix = get_suffix(adj)

    base = data_path / "portfolio_returns"
    return_path = base / f"{excntry}_{pfs}_{n_buckets}_port_ret_month{suffix}.parquet"

    plot_dir = base_path / "exhibits"
    plot_dir.mkdir(parents=True, exist_ok=True)

    strategy_returns = (
        pl.read_parquet(return_path)
        .filter(
            (pl.col("model") != "ORACLE") &
            (pl.col("model") != "RANDOM")
        )
        .with_columns(pl.col("eom").cast(pl.Date))
        .sort(["model", "bucket", "eom"])
    )

    if adj == 2:
        strategy_returns = strategy_returns.with_columns(
            pl.col("model").str.replace("_CLS$", "").str.replace("^LOGIT$", "OLS")
            .alias("model")
    )

    df = strategy_returns.to_pandas()

    df["eom"] = pd.to_datetime(df["eom"])

    df["model"] = df["model"].replace({"OLS": "Linear"})

    df = (
        df.sort_values(["model", "bucket", "eom"])
        .assign(
            cumulative_return=lambda x:
                x.groupby(["model", "bucket"])["mean_ret_bucket_monthly"]
                 .transform(lambda r: r.cumsum() * 100)
        )
    )

    available_models = df["model"].unique().tolist()
    preferred_order = ["Linear", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "FFNN", "COMB"]
    models = [m for m in preferred_order if m in available_models]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # If 7 models: use 3 plots on top and 4 below
    if len(models) == 7:
        plot_axes = [
            axes[0], axes[1], axes[2],
            axes[4], axes[5], axes[6], axes[7]
        ]
    else:
        plot_axes = axes[:len(models)]

    for ax, model in zip(plot_axes, models):

        df_model = df[df["model"] == model]

        for bucket in sorted(df_model["bucket"].unique()):

            tmp = df_model[df_model["bucket"] == bucket]

            ax.plot(
                tmp["eom"],
                tmp["cumulative_return"],
                linewidth=1.8,
                alpha=0.75,
                label=f"Bucket {bucket}"
            )

        ax.axhline(0, linewidth=0.9, color="0.75", alpha=1.0)

        ax.set_title(model, fontsize=9, fontweight="bold", pad=0)
        ax.grid(True, color="0.75",linewidth=0.8,alpha=0.8)

        axis_grey = "0.75"

        ax.spines["left"].set_color(axis_grey)
        ax.spines["bottom"].set_color(axis_grey)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        ax.tick_params(
            axis="both",
            colors=axis_grey,
            labelcolor="black",
            width=1.0
        )

        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    used_axes = set(plot_axes)

    for ax in axes:
        if ax not in used_axes:
            ax.set_visible(False)

    fig.text(
        0.015, 0.5,
        "Cumulative Return(%)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels) // 2,
        frameon=False,
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.12,
        top=0.94,
        wspace=0.07,
        hspace=0.11
    )
    out_path = None

    if save:
        out_path = (
            plot_dir
            / f"{excntry}_{pfs}_{n_buckets}_cum_perf{suffix}.png"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_cum_perf_hml(
        base_path,
        data_path,
        excntry,
        pfs,
        n_buckets,
        adjs,
        adj_labels=None,
        save=True,
        show=False
):



    base = data_path / "portfolio_returns"

    plot_dir = base_path / "exhibits"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for adj in adjs:

        suffix = get_suffix(adj)

        return_path = (
            base
            / f"{excntry}_{pfs}_{n_buckets}_port_ret_month{suffix}.parquet"
        )

        tmp = (
            pl.read_parquet(return_path)
            .filter(
                (pl.col("model") != "ORACLE") &
                (pl.col("model") != "RANDOM")
            )
            .with_columns([
                pl.col("eom").cast(pl.Date),
                pl.lit(adj).alias("adj"),
                pl.lit(adj_labels.get(adj, f"Adj. {adj}")).alias("adjustment")
            ])
        )

        if adj == 2:
            tmp = tmp.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )

        tmp = (
            tmp
            .group_by(["eom", "model", "adj", "adjustment"])
            .agg(
                pl.first("hml_ret_monthly").alias("hml_ret_monthly")
            )
            .sort(["model", "adjustment", "eom"])
        )

        dfs.append(tmp)

    strategy_returns = pl.concat(dfs)

    df = strategy_returns.to_pandas()

    df["eom"] = pd.to_datetime(df["eom"])

    df["model"] = df["model"].replace({"OLS": "Linear"})

    df = (
        df.sort_values(["model", "adj", "eom"])
        .assign(
            cumulative_return=lambda x:
                x.groupby(["model", "adj"])["hml_ret_monthly"]
                 .transform(lambda r: r.cumsum() * 100)
        )
    )

    available_models = df["model"].unique().tolist()

    preferred_order = [
        "Linear", "PLS", "LASSO", "ENET",
        "RF", "GBRT", "XGB", "FFNN", "COMB"
    ]

    models = [m for m in preferred_order if m in available_models]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(12, 6),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    if len(models) == 7:
        plot_axes = [
            axes[0], axes[1], axes[2],
            axes[4], axes[5], axes[6], axes[7]
        ]
    else:
        plot_axes = axes[:len(models)]

    axis_grey = "0.75"

    for ax, model in zip(plot_axes, models):

        df_model = df[df["model"] == model]

        for adj in adjs:

            tmp = df_model[df_model["adj"] == adj]

            if tmp.empty:
                continue

            ax.plot(
                tmp["eom"],
                tmp["cumulative_return"],
                linewidth=1.8,
                alpha=0.80,
                label=adj_labels.get(adj, f"Adj. {adj}")
            )

        ax.axhline(0, linewidth=0.9, color="0.75", alpha=1.0)

        ax.set_title(model, fontsize=9, fontweight="bold", pad=0)

        ax.grid(
            True,
            color="0.75",
            linewidth=0.8,
            alpha=0.8
        )

        ax.spines["left"].set_color(axis_grey)
        ax.spines["bottom"].set_color(axis_grey)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            axis="both",
            colors=axis_grey,
            labelcolor="black",
            width=1.0
        )

        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    used_axes = set(plot_axes)

    for ax in axes:
        if ax not in used_axes:
            ax.set_visible(False)

    fig.text(
        0.015,
        0.5,
        "Cumulative Return (%)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    handles, labels = plot_axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.12,
        top=0.94,
        wspace=0.07,
        hspace=0.11
    )

    out_path = None

    if save:
        out_path = (
            plot_dir
            / f"{excntry}_{pfs}_{n_buckets}_cum_perf_hml.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path

def latex_strat_sent_alphas(
    dfs,
    adjs,
    panel_titles,
    period_order=("Bull", "Bear"),
    panels_per_table: int = 2,
    caption: str = "Statistical Significance of Machine Learning Factor Portfolio Returns in Bull and Bear Periods",
    label: str = "tab:strat-alphas-bull-bear-panels",
):
    model_order = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "COMB", "ORACLE"]

    header_labels = {
        "OLS": "Linear",
        "PLS": "PLS",
        "LASSO": "LASSO",
        "ENET": "ENET",
        "RF": "RF",
        "GBRT": "GBRT",
        "XGB": "XGB",
        "COMB": "COMB",
        "ORACLE": "ORACLE",
    }

    row_specs = [
        ("alpha_mean", "tstat_mean", "High-Low"),
        ("alpha_ff", "tstat_ff", r"$\alpha_{F6}$"),
        ("alpha_ew", "tstat_ew", r"$\alpha_{EW}$"),
        ("alpha_random", "tstat_random", r"$\alpha_{\mathrm{R}}$"),
    ]

    notes = (
        r"This table reports monthly returns and alphas on High--Low portfolios of factor "
        r"strategies formed on the predictions of machine learning models, separately for "
        r"Bull and Bear periods. Bull and Bear periods are defined as months with high and "
        r"low investor sentiment, respectively, based on whether the Baker and Wurgler "
        r"(2006) sentiment index is above or below its median level. The considered models "
        r"include OLS or LOGIT (Linear), PLS, LASSO, ENET, random forests (RF), GBRT, "
        r"XGBoost (XGB), forecast combination (COMB), and a perfect-foresight model "
        r"(ORACLE). High--Low is the spread portfolio that assumes a long position in the "
        r"decile of factors with the highest predicted value and a short position in the "
        r"decile of factors with the lowest predicted value. "
        r"$\alpha_{F6}$, $\alpha_{EW}$, and $\alpha_{\mathrm{R}}$ are alphas from the "
        r"Fama--French six-factor model, the equal-weighted factor benchmark model, and "
        r"the random-sorting benchmark model, respectively. All returns and alphas are "
        r"expressed in percentages. The numbers in parentheses are Newey--West adjusted "
        r"t-statistics with six lags. The table reports results for four target "
        r"transformations: next-month value-weighted excess factor returns (Benchmark), "
        r"cross-sectionally demeaned next-month factor returns (Cross-Reg), a binary "
        r"outperformer indicator based on cross-sectionally demeaned returns (Cross-Class), "
        r"and each factor's percentile rank within the monthly cross-section (Rank-Reg). "
        r"COMB is the equal-weighted average forecast across available models; for "
        r"Cross-Class, it averages predicted outperformance probabilities. ORACLE sorts "
        r"factors on realized next-month returns and therefore represents an ex-post "
        r"perfect-foresight benchmark that is not implementable in real time. The sample "
        r"comprises 153 factors from Jensen, Kelly, and Pedersen (2023); the forecasting "
        r"sample runs from October 1971 to September 2024, and the out-of-sample testing "
        r"period runs from October 1986 to September 2024."
    )

    def fmt_pct(x):
        if x is None:
            return "-"
        return f"{float(x) * 100:.2f}"

    def fmt_t(x):
        if x is None:
            return "-"
        return f"({float(x):.2f})"

    def clean_model_names(df, adj):
        if adj == 2:
            df = df.with_columns(
                pl.col("model")
                .str.replace("_CLS$", "")
                .str.replace("^LOGIT$", "OLS")
                .alias("model")
            )
        return df

    def build_period_rows(row_map, period_label):
        rows = []

        for j, (value_col, tstat_col, row_label) in enumerate(row_specs):
            value_cells = []
            tstat_cells = []

            for model in model_order:
                row = row_map.get((model, period_label), None)

                if row is None:
                    value_cells.append("-")
                    tstat_cells.append("-")
                else:
                    value_cells.append(fmt_pct(row.get(value_col)))
                    tstat_cells.append(fmt_t(row.get(tstat_col)))

            if j > 0:
                rows.append(r"\addlinespace[0.4em]")

            rows.append(row_label + " & " + " & ".join(value_cells) + r" \\")
            rows.append(" & " + " & ".join(tstat_cells) + r" \\")

        return rows

    def build_panel_rows(df, adj, panel_title):
        df = clean_model_names(df.clone(), adj)

        row_map = {
            (row["model"], row["period"]): row
            for row in df.iter_rows(named=True)
        }

        rows = []

        for p, period_label in enumerate(period_order):
            if p > 0:
                rows.append(r"\addlinespace[0.5em]")

            rows.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{panel_title} -- {period_label}}}}} \\"
            )

            rows.extend(build_period_rows(row_map, period_label))

        return rows

    def add_notes(lines):
        lines.extend([
            r"\bottomrule",
            "",
            r"\addlinespace[0.3em]",
            rf"\multicolumn{{{n_cols}}}{{p{{\dimexpr\textwidth-2\tabcolsep\relax}}}}{{",
            r"\scriptsize",
            rf"\textbf{{Notes:}} {notes}}}\\",
        ])

    def chunks(x, n):
        for start in range(0, len(x), n):
            yield start, x[start:start + n]

    col_spec = "C{1.8cm} " + " ".join(["Z"] * len(model_order))
    n_cols = 1 + len(model_order)

    table_blocks = []
    zipped_panels = list(zip(dfs, adjs, panel_titles))

    for chunk_start, panel_chunk in chunks(zipped_panels, panels_per_table):
        table_index = chunk_start // panels_per_table
        is_first_table = table_index == 0

        if is_first_table:
            this_caption = caption
            this_label = label
        else:
            this_caption = rf"{caption} \textit{{(continued)}}"
            this_label = f"{label}-continued-{table_index + 1}"

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{{this_caption}}}",
            rf"\label{{{this_label}}}",
            r"\scriptsize",
            r"\renewcommand{\arraystretch}{1.0}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
            r"\toprule",
            " & " + " & ".join(
                rf"\textbf{{{header_labels[model]}}}" for model in model_order
            ) + r" \\",
            r"\midrule",
        ]

        for i, (df, adj, title) in enumerate(panel_chunk):
            if i > 0:
                lines.append(r"\addlinespace[0.5em]")
                lines.append(r"\midrule")

            lines.extend(build_panel_rows(df, adj, title))

        add_notes(lines)

        lines.extend([
            r"\end{tabularx}",
            "",
            r"\end{table}",
        ])

        table_blocks.append("\n".join(lines))

        if chunk_start + panels_per_table < len(zipped_panels):
            table_blocks.append(r"\clearpage")

    return "\n\n".join(table_blocks)


def plot_variable_importance(
        base_path,
        data_path,
        excntry,
        pfs,
        adj,
        top_n=10,
        save=True,
        show=False):


    suffix = get_suffix(adj)

    base = data_path / "ml_model_output"
    vi_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"

    plot_dir = base_path / "exhibits"
    plot_dir.mkdir(parents=True, exist_ok=True)

    vi = (
        pl.read_parquet(vi_path)
        .filter(
            (pl.col("country") == excntry) &
            (pl.col("pfs") == pfs)
        )
    )

    if adj == 2:
        vi = vi.with_columns(
            pl.col("model")
            .str.replace("_CLS$", "")
            .str.replace("^LOGIT$", "OLS")
            .alias("model")
        )

    vi = (
        vi
        .filter(
            (pl.col("model") != "ORACLE") &
            (pl.col("model") != "RANDOM")
        )
        .group_by(["model", "feature"])
        .agg(
            pl.col("importance").mean().alias("importance")
        )
        .with_columns(
            pl.col("model")
            .str.replace("^OLS$", "Linear")
            .alias("model")
        )
    )

    available_models = vi.select("model").unique()["model"].to_list()

    preferred_order = [
        "Linear", "PLS", "LASSO", "ENET",
        "RF", "GBRT", "XGB", "FFNN", "COMB"
    ]

    models = [m for m in preferred_order if m in available_models]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(9.4, 5.6)
    )

    axes = axes.flatten()

    if len(models) == 7:
        plot_axes = [
            axes[0], axes[1], axes[2],
            axes[4], axes[5], axes[6], axes[7]
        ]
    else:
        plot_axes = axes[:min(len(models), len(axes))]

    bar_color = "#6BAED6"

    for ax, model in zip(plot_axes, models):

        plot_df = (
            vi
            .filter(pl.col("model") == model)
            .sort("importance", descending=True)
            .head(top_n)
            .sort("importance")
        )

        if plot_df.height == 0:
            ax.set_visible(False)
            continue

        df_plot = plot_df.to_pandas()

        ax.barh(
            df_plot["feature"],
            df_plot["importance"],
            color=bar_color,
            edgecolor=bar_color,
            height=0.65
        )

        max_imp = float(df_plot["importance"].max())
        
        if max_imp <= 0.05:
            xmax = 0.05
            xticks = [0.00, 0.025, 0.05]
        elif max_imp <= 0.10:
            xmax = 0.10
            xticks = [0.00, 0.05, 0.10]
        elif max_imp <= 0.15:
            xmax = 0.15
            xticks = [0.00, 0.05, 0.10, 0.15]
        elif max_imp <= 0.20:
            xmax = 0.20
            xticks = [0.00, 0.05, 0.10, 0.15, 0.20]
        elif max_imp <= 0.30:
            xmax = 0.30
            xticks = [0.00, 0.10, 0.20, 0.30]
        elif max_imp <= 0.40:
            xmax = 0.40
            xticks = [0.00, 0.10, 0.20, 0.30, 0.40]
        elif max_imp <= 0.60:
            xmax = 0.60
            xticks = [0.00, 0.20, 0.40, 0.60]
        else:
            xmax = round(max_imp * 1.15, 1)
            xticks = [x / 10 for x in range(0, int(xmax * 10) + 1, 2)]

        ax.set_xlim(0, xmax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.2f}" for x in xticks], fontsize=8)

        ax.set_title(
            model,
            fontsize=7,
            fontweight="bold",
            pad=4
        )

        ax.tick_params(axis="y", labelsize=7, length=0)
        ax.tick_params(axis="x", labelsize=8, length=3, pad=1)

        ax.grid(
            axis="x",
            linestyle="--",
            linewidth=0.8,
            alpha=0.55
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

        ax.set_xlabel("")
        ax.set_ylabel("")

    used_axes = set(plot_axes)

    for ax in axes:
        if ax not in used_axes:
            ax.set_visible(False)

    fig.subplots_adjust(
        left=0.125,
        right=0.970,
        top=0.810,
        bottom=0.080,
        wspace=1.15,
        hspace=0.42
    )


    out_path = None

    if save:
        out_path = (
            plot_dir
            / f"{excntry}_{pfs}_var_imp{suffix}.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_agg_variable_importance_old(
        base_path,
        data_path,
        excntry,
        pfs,
        adj,
        top_ks=(3, 5, 10),
        save=True,
        show=False):


    suffix = get_suffix(adj)

    base = data_path / "ml_model_output"
    vi_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"

    plot_dir = base_path / "exhibits"
    plot_dir.mkdir(parents=True, exist_ok=True)

    vi = (
        pl.read_parquet(vi_path)
        .filter(
            (pl.col("country") == excntry) &
            (pl.col("pfs") == pfs)
        )
    )

    if adj == 2:
        vi = vi.with_columns(
            pl.col("model")
            .str.replace("_CLS$", "")
            .str.replace("^LOGIT$", "OLS")
            .alias("model")
        )

    vi = (
        vi
        .filter(
            (pl.col("model") != "ORACLE") &
            (pl.col("model") != "RANDOM")
        )
        .group_by(["model", "feature"])
        .agg(
            pl.col("importance").mean().alias("importance")
        )
    )

    available_models = vi.select("model").unique()["model"].to_list()

    preferred_order = [
        "OLS", "PLS", "LASSO", "ENET",
        "RF", "GBRT", "XGB", "FFNN", "COMB"
    ]

    models = [m for m in preferred_order if m in available_models]

    if len(models) == 0:
        raise ValueError(f"No valid models found in {vi_path}")

    rows = []

    for model in models:

        model_vi = (
            vi
            .filter(pl.col("model") == model)
            .sort("importance", descending=True)
        )

        for k in top_ks:

            agg_imp = (
                model_vi
                .head(k)
                .select(pl.col("importance").sum())
                .item()
            )

            rows.append({
                "model": model,
                "top_k": f"Top {k}",
                "k": k,
                "importance": agg_imp
            })

    df = pl.DataFrame(rows).to_pandas()

    fig, ax = plt.subplots(
        figsize=(7.2, 4.4)
    )

    x = list(range(len(models)))
    n_groups = len(top_ks)
    bar_width = 0.22

    colors = {
        3: "#BDD7E7",
        5: "#6BAED6",
        10: "#2171B5"
    }

    for i, k in enumerate(top_ks):

        tmp = df[df["k"] == k].set_index("model").loc[models].reset_index()

        offset = (i - (n_groups - 1) / 2) * bar_width

        bars = ax.bar(
            [v + offset for v in x],
            tmp["importance"],
            width=bar_width,
            color=colors.get(k, None),
            label=f"Top {k}"
        )

        for bar, value in zip(bars, tmp["importance"]):

            label = f"{value * 100:.0f}%"

            if value >= 0.25:
                text_color = "white"
                y_pos = value - 0.025
                va = "top"
            else:
                text_color = "black"
                y_pos = value + 0.012
                va = "bottom"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                label,
                ha="center",
                va=va,
                fontsize=9,
                color=text_color
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)

    ax.set_ylabel(
        "Aggregate Contribution Importance",
        fontsize=9,
        fontweight="bold"
    )

    ax.set_ylim(0, max(df["importance"].max() * 1.18, 0.10))

    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.grid(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(top_ks),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.93,
        bottom=0.22
    )

    out_path = None

    if save:
        out_path = (
            plot_dir
            / f"{excntry}_{pfs}_agg_var_imp{suffix}.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path

def plot_agg_variable_importance(
        base_path,
        data_path,
        excntry,
        pfs,
        adj,
        top_ks=(3, 5, 10),
        save=True,
        show=False):


    suffix = get_suffix(adj)

    base = data_path / "ml_model_output"
    vi_path = base / f"{excntry}_{pfs}_vi_month{suffix}.parquet"

    plot_dir = base_path / "exhibits"
    plot_dir.mkdir(parents=True, exist_ok=True)

    vi = (
        pl.read_parquet(vi_path)
        .filter(
            (pl.col("country") == excntry) &
            (pl.col("pfs") == pfs)
        )
    )

    if adj == 2:
        vi = vi.with_columns(
            pl.col("model")
            .str.replace("_CLS$", "")
            .str.replace("^LOGIT$", "OLS")
            .alias("model")
        )

    vi = (
        vi
        .filter(
            (pl.col("model") != "ORACLE") &
            (pl.col("model") != "RANDOM")
        )
        .group_by(["model", "feature"])
        .agg(
            pl.col("importance").mean().alias("importance")
        )
    )

    available_models = vi.select("model").unique()["model"].to_list()

    preferred_order = [
        "OLS", "PLS", "LASSO", "ENET",
        "RF", "GBRT", "XGB", "FFNN", "COMB"
    ]

    models = [m for m in preferred_order if m in available_models]

    if len(models) == 0:
        raise ValueError(f"No valid models found in {vi_path}")

    rows = []

    for model in models:

        model_vi = (
            vi
            .filter(pl.col("model") == model)
            .sort("importance", descending=True)
        )

        for k in top_ks:

            agg_imp = (
                model_vi
                .head(k)
                .select(pl.col("importance").sum())
                .item()
            )

            rows.append({
                "model": model,
                "top_k": f"Top {k}",
                "k": k,
                "importance": agg_imp
            })

    df = pl.DataFrame(rows).to_pandas()

    fig, ax = plt.subplots(figsize=(10, 4))

    x = list(range(len(models)))
    n_groups = len(top_ks)
    bar_width = 0.29

    colors = {
        3: "#BDD7E7",
        5: "#6BAED6",
        10: "#2171B5"
    }

    max_y = df["importance"].max()
    y_offset = max_y * 0.025

    for i, k in enumerate(top_ks):

        tmp = df[df["k"] == k].set_index("model").loc[models].reset_index()

        offset = (i - (n_groups - 1) / 2) * bar_width

        bars = ax.bar(
            [v + offset for v in x],
            tmp["importance"],
            width=bar_width,
            color=colors.get(k, None),
            label=f"Top {k}",
            zorder=2
        )

        for bar, value in zip(bars, tmp["importance"]):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + y_offset,
                f"{value * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
                zorder=3
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)

    for i in range(len(models) - 1):
        ax.axvline(
            x=i + 0.5,
            ymin=0.00,
            ymax=0.92,
            color="0.75",
            linewidth=0.8,
            linestyle="-",
            zorder=0
        )

    ax.set_ylim(0, max_y * 1.18)

    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.grid(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=len(top_ks),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=9
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.22
    )

    out_path = None

    if save:
        out_path = (
            plot_dir
            / f"{excntry}_{pfs}_agg_var_imp{suffix}.png"
        )

        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path