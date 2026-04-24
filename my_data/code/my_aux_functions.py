import time
import datetime
from datetime import date
from pathlib import Path
import os
import duckdb
import pandas as pd
import polars as pl
from tqdm import tqdm
from sktime.split import ExpandingWindowSplitter
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
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

def setup_folder_structure(data_path):
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


    countries = ['USA', 'JPN', 'BRA', 'CHN', 'GBR', 'DEU', 'IND']

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

        market_partial.write_parquet(
            data_path / "factor_returns" / f"temp_market_{f.stem}.parquet"
        )
        
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
            
            # winsorize
            sub = sub.with_columns(pl.col("var").clip(
                lower_bound=pl.col("var").quantile(0.01).over("eom"),
                upper_bound=pl.col("var").quantile(0.99).over("eom"))
            )
            sub = sub.with_columns([
                pl.col(c).clip(
                lower_bound=pl.col(c).quantile(0.01).over("eom"),
                upper_bound=pl.col(c).quantile(0.99).over("eom")).alias(c)for c in chars])

            # create buckets
            sub = add_ecdf(sub)

            sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("cdf_min"))
            sub = sub.with_columns(
                pl.when(pl.col("cdf") == pl.col("cdf_min"))
                .then(0.00000001)
                .otherwise(pl.col("cdf"))
                .alias("cdf")
            )
            # turn into 3 portfolios
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
        pl.scan_parquet(data_path / "factor_returns" / "temp_market_*.parquet")
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
                .otherwise(-1)
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

        spreads = lms_returns.select(["eom", "characteristics"] + [f"spread_{c}" for c in chars])
        print(spreads.head())

        df = df.join(spreads, on=["eom", "characteristics"], how="left")
        print(df.describe())

        feature_clip_cols = [
            c for c in df.columns if c.startswith(("ret_m_", "ret_y_", "vol_m_", "beta_", "spread_"))]
        
        df = df.with_columns([
            pl.col(c).clip(
                lower_bound=pl.col(c).quantile(0.01).over("eom"),
                upper_bound=pl.col(c).quantile(0.99).over("eom")
                ).alias(c)
                for c in feature_clip_cols
                ])

        to_clean_cols = [
            c for c in df.columns if (c.startswith(("ret", "vol_m_", "beta_", "spread_")))]
        
        df = df.filter(
            pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in to_clean_cols]))
        
        df = df.sort(["eom", "characteristics"])

        df = df.with_columns(pl.arange(0, pl.len()).alias("row_id"))

        meta = df.select(["row_id", "eom", "characteristics", "excntry"]).sort("row_id")

        X = df.drop(cols_to_drop).sort("row_id")

        if adj == 2:
            y = df.select(["row_id", "target_class"]).sort("row_id")
        else:
            y = df.select(["row_id", base_ret]).sort("row_id")

        X.write_parquet(out_X)
        y.write_parquet(out_y)
        meta.write_parquet(meta_out)


def build_train_val_test_idx(data_path, pfs, adj, excntry, min_train_val, val, test_range, forward_steps):

    test = (test_range * 12)

    if adj == 1:
        suffix = "_cross"
    elif adj == 2:
        suffix = "_class"
    else:
        suffix = ""

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
        print(f"train starts from: {train_months.min()} to {train_months.max()}/n")
        val_months = unique_eom.iloc[val_months_idx]
        print(f"val starts from: {val_months.min()} to {val_months.max()}/n")
        test_months = unique_eom.iloc[test_months_idx]
        print(f"test starts from: {test_months.min()} to {test_months.max()}/n")

        train_rows = meta.loc[meta["eom"].isin(train_months), "row_id"].values
        val_rows = meta.loc[meta["eom"].isin(val_months), "row_id"].values
        test_rows = meta.loc[meta["eom"].isin(test_months), "row_id"].values
        

        expand_wind_splits.append({
            "train" : train_rows,
            "val" : val_rows,
            "test" : test_rows
        })
        
    return expand_wind_splits


def predict_with_ols(X_train, y_train, X_test, y_test, r2_split, y_pred_split, y_true_split):

    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"ols - r2_pred: {r2} for this split")

    r2_split.setdefault("OLS", []).append(r2)
    y_pred_split.setdefault("OLS", []).append(y_pred.ravel())
    y_true_split.setdefault("OLS", []).append(y_test.values.ravel())

    return lr, None, r2

def tune_model_with_val(model, param_dist, X_train, y_train, X_val, y_val, n_iter=15):
    
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


def predict_with_pls(X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,
                     r2_split, y_pred_split, y_true_split):
                     
    param_dist = {"n_components": list(range(1, 11))}
    
    best_params = tune_model_with_val(
        PLSRegression(max_iter=5000), param_dist, X_train, y_train, X_val, y_val, n_iter=10)
    
    print(f"PLS best params: {best_params}")

    best_model = PLSRegression(max_iter=5000, **best_params)
    
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"pls - r2_pred: {r2} for this split")

    r2_split.setdefault("PLS", []).append(r2)
    y_pred_split.setdefault("PLS", []).append(y_pred.ravel())
    y_true_split.setdefault("PLS", []).append(y_test.values.ravel())

    return best_model, best_params, r2

def predict_with_lasso(X_train,y_train, X_val, y_val, X_train_val, y_train_val,
                       X_test, y_test, r2_split, y_pred_split, y_true_split):

    
    param_dist = {"alpha": np.logspace(-3, np.log10(0.002), 100)}

    best_params = tune_model_with_val(
            Lasso(max_iter=5000), param_dist, X_train, y_train, X_val, y_val, n_iter=20)
    print(f"Lasso best params: {best_params}")

    best_model = Lasso(max_iter=5000, **best_params)

    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    n_total = len(best_model.coef_)
    n_zero = (best_model.coef_ == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    r2 = r2_score(y_test, y_pred)
    print(f"lasso - r2_pred: {r2} for this split")


    r2_split.setdefault("LASSO", []).append(r2)
    y_pred_split.setdefault("LASSO", []).append(y_pred.ravel())
    y_true_split.setdefault("LASSO", []).append(y_test.values.ravel())

    return best_model, best_params, r2

def predict_with_enet(X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,
                      r2_split, y_pred_split, y_true_split):
                      
    param_dist = {
        "alpha": np.logspace(-3, np.log10(0.003), 50)}

    best_params = tune_model_with_val(
            ElasticNet(max_iter=5000, l1_ratio=0.5), param_dist, X_train, y_train, X_val, y_val, n_iter=20)

    print(f"ENET best params: {best_params}")

    best_model = ElasticNet(max_iter=5000, l1_ratio=0.5, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    n_total = len(best_model.coef_)
    n_zero = (best_model.coef_ == 0).sum()
    print(f"Zero coefficients: {n_zero}/{n_total}")

    r2 = r2_score(y_test, y_pred)
    print(f"enet - r2_pred: {r2} for this split")

    r2_split.setdefault("ENET", []).append(r2)
    y_pred_split.setdefault("ENET", []).append(y_pred.ravel())
    y_true_split.setdefault("ENET", []).append(y_test.values.ravel())

    return best_model, best_params, r2


def predict_with_rf(X_train, y_train, X_val, y_val, X_train_val, y_train_val,
                    X_test, y_test, r2_split, y_pred_split, y_true_split):

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10],
        "max_features": ["sqrt", 0.3]
    }

    best_params = tune_model_with_val(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, X_train, y_train, X_val, y_val, n_iter=10)

    print(f"RF best params: {best_params}")

    best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"rf - r2_pred: {r2} for this split")

    r2_split.setdefault("RF", []).append(r2)
    y_pred_split.setdefault("RF", []).append(y_pred.ravel())
    y_true_split.setdefault("RF", []).append(y_test.values.ravel())

    return best_model, best_params, r2

def predict_with_gbrt(
        X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test,
          r2_split, y_pred_split, y_true_split):
    
    param_dist = {
        "learning_rate": [0.03, 0.05],
        "max_depth": [5, 7],
        "min_samples_leaf": [10, 20],
        "max_iter": [100, 200]
    }

    best_params = tune_model_with_val(
        HistGradientBoostingRegressor(random_state=42),
        param_dist,
        X_train, y_train,
        X_val, y_val,
        n_iter=8
        
    )
    print(f"GBRT best params: {best_params}")

    best_model =  HistGradientBoostingRegressor(random_state=42, **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"gbrt - r2_pred: {r2} for this split")

    r2_split.setdefault("GBRT", []).append(r2)
    y_pred_split.setdefault("GBRT", []).append(y_pred.ravel())
    y_true_split.setdefault("GBRT", []).append(y_test.values.ravel())

    return best_model, best_params, r2


def predict_with_xgb(
    X_train, y_train,
    X_val, y_val,
    X_train_val, y_train_val,
    X_test, y_test,
    r2_split, y_pred_split, y_true_split):
    
    best_score = -np.inf
    best_params = None
    best_iteration = None

    for lr in [0.03, 0.1]:
        for max_d in [5, 7]:
            for subs in [0.6, 0.8]:
                for colsample in [0.6, 0.8]:
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

    final_model = XGBRegressor(
        **best_params,
        n_estimators=best_iteration, 
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train_val, y_train_val, verbose=False)

    y_pred = final_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"xgb - r2_pred: {r2} for this split")

    r2_split.setdefault("XGB", []).append(r2)
    y_pred_split.setdefault("XGB", []).append(y_pred.ravel())
    y_true_split.setdefault("XGB", []).append(y_test.values.ravel())

    return final_model, best_params, r2

def predict_with_ffnn(X_train, y_train, X_val, y_val, X_train_val, y_train_val,
                       X_test, y_test, r2_split, y_pred_split, y_true_split):
    
    param_dist = {
    "learning_rate_init": [0.0003, 0.001],
    "alpha": [0.001, 0.002, 0.01, 0.05]}

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
        n_iter=4,
        
    )

    print(f"FFNN best params: {best_params}")

    best_model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        **best_params)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"ffnn - r2_pred: {r2} for this split")

    r2_split.setdefault("FFNN", []).append(r2)
    y_pred_split.setdefault("FFNN", []).append(y_pred.ravel())
    y_true_split.setdefault("FFNN", []).append(y_test.values.ravel())

    return best_model, best_params, r2

def train_pred_model(data_path, excntry, pfs, splits_idx, model, adj):

    suffix = "_cross" if adj == 1 else ""
    base_ret = "ret_exc_cross_lead1m_vw_cap" if adj == 1 else "ret_exc_lead1m_vw_cap"
    base = data_path/"factor_characteristics"

    X_path = base /f"{excntry}_{pfs}_feature{suffix}.parquet"
    y_path = base /f"{excntry}_{pfs}_target{suffix}.parquet"

    X_raw = pl.read_parquet(X_path).to_pandas()
    y_raw = pl.read_parquet(y_path).to_pandas()

    X = X_raw.set_index("row_id")
    y = y_raw.set_index("row_id")

    y_series = y[base_ret]

    r2_split = {}
    y_pred_split = {}
    y_true_split = {}
    vi_split = {}
    test_idx_split = []

    if model == "all":
        models_to_run = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "XGB", "FFNN"]
    elif isinstance(model, str):
        models_to_run = [model.upper()]
    else:
        models_to_run = [m.upper() for m in model]

    for split in splits_idx:

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
        
        if "OLS" in models_to_run:
            
            model_ols, best_params_ols, r2_ols = predict_with_ols(
            X_train=X_train_val, y_train=y_train_val,
            X_test=X_test, y_test=y_test,
            r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)

            result = permutation_importance(
                model_ols, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_ols = result.importances_mean
            vi_split.setdefault("OLS", []).append(vi_ols)


        if "PLS" in models_to_run:

            model_pls, best_params_pls, r2_pls = predict_with_pls(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_pls, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_pls = result.importances_mean
            vi_split.setdefault("PLS", []).append(vi_pls)


        if "LASSO" in models_to_run:

            model_lasso, best_params_lasso, r2_lasso = predict_with_lasso(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_lasso, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_lasso = result.importances_mean
            vi_split.setdefault("LASSO", []).append(vi_lasso)


        if "ENET" in models_to_run:

            model_enet, best_params_enet, r2_enet = predict_with_enet(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_enet, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_enet = result.importances_mean
            vi_split.setdefault("ENET", []).append(vi_enet)

        if "RF" in models_to_run:

            model_rf, best_params_rf, r2_rf = predict_with_rf(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_rf, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_rf = result.importances_mean
            vi_split.setdefault("RF", []).append(vi_rf)


        if "GBRT" in models_to_run:

            model_gbrt, best_params_gbrt, r2_gbrt = predict_with_gbrt(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_gbrt, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_gbrt = result.importances_mean
            vi_split.setdefault("GBRT", []).append(vi_gbrt)


        if "FFNN" in models_to_run:

            model_ffnn, best_params_ffnn, r2_ffnn = predict_with_ffnn(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_ffnn, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_ffnn = result.importances_mean
            vi_split.setdefault("FFNN", []).append(vi_ffnn)


        if "XGB" in models_to_run:

            model_xgb, best_params_xgb, r2_xgb = predict_with_xgb(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split
            )

            result = permutation_importance(
                model_xgb, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=42)

            vi_xgb = result.importances_mean
            vi_split.setdefault("XGB", []).append(vi_xgb)

    return r2_split, y_pred_split, y_true_split, test_idx_split, vi_split

def eval_correlation(df, excntry, pfs, adj, data_path):

    if adj == 1:
        suffix = "_cross"
    elif adj == 2:
        suffix = "_class"
    else:
        suffix = ""
    meta = pl.read_parquet(
        data_path/"factor_characteristics"/f"{excntry}_{pfs}_meta{suffix}.parquet").to_pandas()
    
    df = df.merge(meta[["row_id", "eom"]], on="row_id", how="left")
    pear_lst = []
    spear_lst = []
    for _, month in df.groupby("eom"):

        y = month["actual"]
        x = month["prediction"]
        
        if x.std() == 0 or y.std() == 0:
            print(month)
            continue

        pear_lst.append(x.corr(y))
        spear_lst.append(x.corr(y, method = 'spearman'))
    
    pearson = np.nanmean(pear_lst)
    spearman = np.nanmean(spear_lst)
        
    return {"pearson": pearson, "spearman": spearman}


def eval_model(
        data_path, pfs, r2_split, y_pred_split, y_true_split, test_idx_split, vi_split, excntry, adj):

    suffix = "_cross" if adj == 1 else ""
    base = data_path / "ml_model_output"
    
    meta = pl.read_parquet(
        data_path / "factor_characteristics" / f"{excntry}_{pfs}_meta{suffix}.parquet"
        ).select(["row_id", "eom"])
    
    X_path = data_path/"factor_characteristics"/f"{excntry}_{pfs}_feature{suffix}.parquet"
    vi_path = base / f"{excntry}_{pfs}_vi{suffix}.parquet"
    vi_ts_path = base / f"{excntry}_{pfs}_vi_series{suffix}.parquet"
    results_monthly_path = base /f"{excntry}_{pfs}_ml_models_monthly{suffix}.parquet"
    results_global_path = base / f"{excntry}_{pfs}_ml_models_global{suffix}.parquet"
    
    y_path_raw = data_path/"factor_characteristics"/f"{excntry}_{pfs}_target.parquet"
    y_raw = pl.read_parquet(y_path_raw)
    y_raw = y_raw.rename({"ret_exc_lead1m_vw_cap": "ret_raw"})
    y_raw = y_raw.select(["row_id", "ret_raw"])

    feature_names = pl.read_parquet(X_path).to_pandas().drop(columns="row_id").columns
    vi_global_lst = []
    vi_ts_lst = []

    for model, vi_list in vi_split.items():

        if model == "COMB":
            continue

        vi_avg = np.mean(vi_list, axis=0)
        vi_avg = np.maximum(vi_avg, 0)

        if vi_avg.sum() != 0:
            vi_norm = vi_avg / vi_avg.sum()
        else:
            vi_norm = vi_avg

        for f, imp in zip(feature_names, vi_norm):
            vi_global_lst.append({
                "feature": f,
                "importance": imp,
                "model": model,
                "country": excntry,
                "pfs": pfs
            })

        for split_no, vi_arr in enumerate(vi_list):

            row_ids = test_idx_split[split_no]

            split_dates = (
                meta
                .filter(pl.col("row_id").is_in(row_ids))
                .select(pl.col("eom"))
            )

            test_start = split_dates["eom"].min()
            test_end = split_dates["eom"].max()

            vi_arr = np.maximum(vi_arr, 0)

            if vi_arr.sum() != 0:
                vi_arr_norm = vi_arr / vi_arr.sum()
            else:
                vi_arr_norm = vi_arr

            for f, imp in zip(feature_names, vi_arr_norm):
                vi_ts_lst.append({
                    "feature": f,
                    "importance": imp,
                    "model": model,
                    "test_start": test_start,
                    "test_end": test_end,
                    "country": excntry,
                    "pfs": pfs
                })


    results_global_lst = []
    results_monthly_lst = []
    test_idx_all = np.concatenate(test_idx_split)

    for model in y_pred_split.keys():

        y_pred_all = np.concatenate(y_pred_split[model])
        y_true_all = np.concatenate(y_true_split[model])
        
        results_monthly = pl.DataFrame({
            "row_id": test_idx_all,
            "prediction": y_pred_all,
            "actual": y_true_all,
            "model": model,
            "country": excntry,
            "pfs": pfs}
        )

        results_monthly = results_monthly.join(y_raw, on="row_id", how="left").to_pandas()
        
        results_monthly_lst.append(results_monthly)

        r2_global = r2_score(y_true_all, y_pred_all)
        out_corr = eval_correlation(df=results_monthly, pfs=pfs, adj=adj, excntry=excntry, data_path=data_path)

        results_global_lst.append({
        "country": excntry,
        "pfs": pfs,
        "model": model,
        "r2": r2_global,
        "spearman": out_corr["spearman"],
        "pearson": out_corr["pearson"]})


    df_vi = pl.DataFrame(vi_global_lst)
    if vi_path.exists():
        existing = pl.read_parquet(vi_path)
        df_vi = pl.concat([existing, df_vi])
        df_vi = df_vi.unique(subset=["model", "country", "pfs"], keep="last")
    df_vi.write_parquet(vi_path)
    print(df_vi.sort(by="importance", descending=True).head(10))

    df_vi_ts = pl.DataFrame(vi_ts_lst)
    if vi_ts_path.exists():
        existing = pl.read_parquet(vi_ts_path)
        df_vi_ts = pl.concat([existing, df_vi_ts])
        df_vi_ts = df_vi_ts.unique(subset=["model", "country", "pfs"], keep="last")
    df_vi_ts.write_parquet(vi_ts_path)

    df_global = pl.from_pandas(pd.DataFrame(results_global_lst))
    if results_global_path.exists():
        existing = pl.read_parquet(results_global_path)
        df_global = pl.concat([existing, df_global])
        df_global = df_global.unique(subset=["model", "country", "pfs"], keep="last")
    df_global.write_parquet(results_global_path)
    print(df_global)

    df_monthly = pl.from_pandas(pd.concat(results_monthly_lst, ignore_index=True))
    if results_monthly_path.exists():
        existing = pl.read_parquet(results_monthly_path)
        df_monthly = pl.concat([existing, df_monthly])
        df_monthly = (df_monthly.sort("row_id").unique(subset=["row_id", "model", "country", "pfs"], keep="last"))
    df_monthly.write_parquet(results_monthly_path)

def add_comb_model(excntry, pfs, data_path, adj):
    
    suffix = "_cross" if adj == 1 else ""
    results_monthly_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_ml_models_monthly{suffix}.parquet"

    monthly_df = pl.read_parquet(results_monthly_path).filter(pl.col("model") != "COMB")

    comb_df = (monthly_df
            .filter((pl.col("model") != "FFNN") & (pl.col("model") != "OLS") & (pl.col("model") != "PLS"))
            .group_by(["row_id", "country", "pfs"])
            .agg([
                pl.col("prediction").mean().alias("prediction"),
                pl.col("actual").first().alias("actual"),
                pl.col("ret_raw").first().alias("ret_raw")
                ]).with_columns(pl.lit("COMB").alias("model")))

    comb_df = comb_df.select(monthly_df.columns)
    monthly_df = pl.concat([monthly_df, comb_df])
    monthly_df.write_parquet(results_monthly_path)

    results_global_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_ml_models_global{suffix}.parquet"
    global_df = pl.read_parquet(results_global_path).filter(pl.col("model") != "COMB")

    r2_global = r2_score(comb_df["actual"].to_numpy(), comb_df["prediction"].to_numpy())
    out_corr = eval_correlation(df=comb_df.to_pandas(),pfs=pfs, adj=adj, excntry=excntry, data_path=data_path)
    comb_global = pl.DataFrame({
        "country": [excntry],
        "pfs": [pfs],
        "model": ["COMB"],
        "r2": [r2_global],
        "spearman": [out_corr["spearman"]],
        "pearson": [out_corr["pearson"]],
    })

    global_df = pl.concat([global_df, comb_global])
    global_df.write_parquet(results_global_path)

    return monthly_df, global_df


def build_strategy_returns(data_path, excntry, pfs, n_buckets, adj):

    meta_path = data_path/"factor_characteristics"/f"{excntry}_meta.parquet"
    
    suffix = "_cross" if adj == 1 else ""
    base = data_path / "portfolio_returns"
    
    model_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_ml_models_monthly{suffix}.parquet"
    
    out_path_mon = base/f"{excntry}_{pfs}_{n_buckets}_port_ret_monthly_avg{suffix}.parquet"
    out_path_buck = base/f"{excntry}_{pfs}_{n_buckets}_factor_ret_to_bucket{suffix}.parquet"
    out_path_gl = base/f"{excntry}_{pfs}_{n_buckets}_port_ret_global_avg{suffix}.parquet"

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

    oracle_buckets, oracle_monthly, oracle_global = run_sort(
        eom_pred, "ret_raw", "ORACLE"
    )

    bucket_lst.append(oracle_buckets)
    monthly_lst.append(oracle_monthly)
    global_lst.append(oracle_global)

    for model in models:

        df_model = eom_pred.filter(pl.col("model") == model)

        ml_buckets, port_ret_monthly, port_ret_global = run_sort(
            df_model, "prediction", model
        )

        bucket_lst.append(ml_buckets)
        monthly_lst.append(port_ret_monthly)
        global_lst.append(port_ret_global)

    all_port_ret_bucket = pl.concat(bucket_lst)
    all_port_ret_monthly = pl.concat(monthly_lst)
    all_port_ret_global = pl.concat(global_lst)

    all_port_ret_monthly.write_parquet(out_path_mon)
    all_port_ret_bucket.write_parquet(out_path_buck)
    all_port_ret_global.write_parquet(out_path_gl)

    return all_port_ret_global


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

        meta_path = data_path/"factor_characteristics"/f"{excntry}_meta.parquet"

        suffix = "_cross" if adj == 1 else ""

        base = data_path / "portfolio_returns"

        bucket_path = base / f"{excntry}_{pfs}_{n_buckets}_factor_ret_to_bucket{suffix}.parquet"
        return_path = base / f"{excntry}_{pfs}_{n_buckets}_port_ret_monthly_avg{suffix}.parquet"
        regressions_strategy_path = base / f"{excntry}_{pfs}_{n_buckets}_regression_strategy{suffix}.parquet"
        regressions_bucket_path = base / f"{excntry}_{pfs}_{n_buckets}_regression_bucket{suffix}.parquet"
        turnover_path = base / f"{excntry}_{pfs}_{n_buckets}_turnover{suffix}.parquet"

        meta_data = pl.read_parquet(meta_path).select(["row_id", "characteristics"])
        factor_to_bucket = pl.read_parquet(bucket_path)
        strategy_returns = pl.read_parquet(return_path)
        ew_strategy_returns = (
                    factor_to_bucket
                    .select(["eom", "row_id", "ret_raw"]).unique()
                    .group_by("eom")
                    .agg(pl.col("ret_raw").mean().alias("ew_strategy_ret")))
        
        ff_monthly = pl.read_parquet(data_path/"other_input"/"ff_monthly.parquet"
                                             ).with_columns(pl.col("eom").cast(pl.Date))
        
        models = strategy_returns.select("model").unique().to_series().to_list()

        regression_lst = []
        regression_bucket_lst = []
        turnover_lst = []
        
        for model in models:
            
            strat_model = strategy_returns.filter(pl.col("model") == model)
            bucket_model = factor_to_bucket.filter(pl.col("model") == model)

            df_regress = (
                strat_model
                .group_by("eom")
                .agg(pl.col("hml_ret_monthly").first().alias("strategy_ret"))
                .join(ew_strategy_returns, on="eom", how="left")
                .join(ff_monthly, on="eom", how="left")
                .sort("eom")
                .to_pandas()
                )
            
            X_ff = df_regress[["mkt", "smb", "hml", "rmw", "cma", "wml"]]
            X_ff = sm.add_constant(X_ff)
            
            X_ew = df_regress[["ew_strategy_ret"]]
            X_ew = sm.add_constant(X_ew)

           
            y = df_regress["strategy_ret"]
            X_mean = pd.DataFrame({"const": 1}, index=range(len(y)))

            regress_mean = sm.OLS(y, X_mean).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

            regress_ff = sm.OLS(y, X_ff).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            
            regress_ew = sm.OLS(y, X_ew).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            
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
                "tstat_ew": regress_ew.tvalues["const"]})
            
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
                .join(ff_monthly, on="eom", how="left")
                .sort("eom")
                .to_pandas()
                )

            for b in range(1, n_buckets + 1):

                df_regress_bucket = df_b[df_b["bucket"] == b]

                y = df_regress_bucket["ret"]
                
                X_ff = df_regress_bucket[["mkt", "smb", "hml", "rmw", "cma", "wml"]]
                X_ff = sm.add_constant(X_ff)

                X_ew = df_regress_bucket[["ew_strategy_ret"]]
                X_ew = sm.add_constant(X_ew)

                regress_ff = sm.OLS(y, X_ff).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
                regress_ew = sm.OLS(y, X_ew).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

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
                    })
                            
            
        all_regressions_strategy = pl.DataFrame(regression_lst)
        all_regressions_strategy.write_parquet(regressions_strategy_path)
        
        all_turnovers = pl.DataFrame(turnover_lst)
        all_turnovers.write_parquet(turnover_path)
        
        all_regressions_bucket = pl.DataFrame(regression_bucket_lst)
        all_regressions_bucket.write_parquet(regressions_bucket_path)




            






