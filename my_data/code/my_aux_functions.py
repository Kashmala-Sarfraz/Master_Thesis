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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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


def get_countries(data_path):
    countries = set()
    for file in os.listdir(os.path.join(data_path, "stock_characteristics")):
        if file.endswith(".parquet"):
            country = file.split("_")[0]
            countries.add(country)
    return sorted(countries)

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
    
    file_path = data_path / "stock_characteristics" / f"{excntry}_*.parquet"
    data = pl.read_parquet(file_path)
    data = data
    
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
    
    market = (data
              .select(["id", "eom", "ret_exc_lead1m", "me_cap"])
              .group_by(["eom"])
              .agg(
                  ((pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / 
                   pl.col("me_cap").sum()
                   ).alias("market_ret_exc_vw_cap")))

    market = market.with_columns(
        pl.col("eom").dt.offset_by("1m").dt.month_end()
        ).select(["eom","market_ret_exc_vw_cap"]) # market returns at date realized

    char_pfs = []
    for _i, x in enumerate(tqdm(chars, desc="Processing chars", unit="char", ncols=80)):
        sub = (data
               .lazy()
               .with_columns(pl.col(x).cast(pl.Float64).alias("var"))
               .filter(pl.col("var").is_not_null())
               .select([
                        "id",
                        "eom",
                        "var",
                        "size_grp",
                        "ret_exc_lead1m",
                        "me",
                        "me_cap",
                        "crsp_exchcd",
                        "comp_exchg",
               ])
               )
        
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

        # create buckets
        if sub.limit(1).collect().height > 0:

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
                    pl.col("var").min().alias("var_min"),
                    pl.col("var").max().alias("var_max"),
                    pl.len().alias("n"),
                    (
                        (pl.col("var") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("chr_vw_cap"),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("ret_exc_lead1m_vw_cap")
                ]
            )
            
            char_pfs.append(pf_returns.collect())


    if char_pfs:
        pf_returns_total = pl.concat(char_pfs)
        pf_returns_total = pf_returns_total.with_columns(
            pl.lit(excntry).str.to_uppercase().alias("excntry"))
        
        hml_returns = pf_returns_total.group_by(["eom", "characteristics", "excntry"]).agg(
            # check if there are rows with missing long or short portfolios
            pl.col("pf").is_in([pfs, 1]).sum().alias("pfs"),
            # calculate characteristic spread
            (pl.col("chr_vw_cap").filter(pl.col("pf") == pfs).first()
            - pl.col("chr_vw_cap").filter(pl.col("pf") == 1).first()).alias("chr_spread"),
            # calculate long short return
            (pl.col("ret_exc_lead1m_vw_cap").filter(pl.col("pf") == pfs).first()
            - pl.col("ret_exc_lead1m_vw_cap").filter(pl.col("pf") == 1).first()).alias("ret_exc_lead1m_vw_cap"),
            # calculate number of stocks in both portfolios combined
            (pl.col("n").filter(pl.col("pf") == pfs).first()
            + pl.col("n").filter(pl.col("pf") == 1).first()).alias("n_stocks"),
            # calculate the min numeber of stock in either of the two portfolios
            (pl.col("n").filter(pl.col("pf").is_in([pfs, 1])).min().alias("n_stocks_min")))
        
        hml_returns = hml_returns.filter(pl.col("pfs") == 2).drop("pfs")
        hml_returns = hml_returns.sort(["characteristics", "eom"])

        lms_returns = char_info.join(hml_returns, on="characteristics", how = "left")
        resign_cols = ["ret_exc_lead1m_vw_cap", "chr_spread"]
        lms_returns = lms_returns.with_columns([
                (pl.col(var) * pl.col("direction")).alias(var) for var in resign_cols])
        

        lms_returns.write_parquet(
            data_path / "factor_returns" / f"{excntry}_{pfs}_lms_returns.parquet")
        hml_returns.write_parquet(
            data_path / "factor_returns" / f"{excntry}_{pfs}_hml_returns.parquet")
        pf_returns_total.write_parquet(
            data_path / "factor_returns" / f"{excntry}_{pfs}_pf_returns.parquet")
        market.write_parquet(
            data_path / "factor_returns" / f"{excntry}_market_returns.parquet")


def build_factor_characteristics(data_path, pfs, excntry):

    factor = data_path / "factor_returns" / f"{excntry}_{pfs}_lms_returns.parquet"
    market = data_path / "factor_returns" / f"{excntry}_market_returns.parquet"

    if (factor.exists() & factor.exists()):
        lms_returns = pl.read_parquet(factor)
        market_returns = pl.read_parquet(market)

        left = (lms_returns
                .select(["eom", "characteristics", "ret_exc_lead1m_vw_cap", "excntry"])
                .sort(["characteristics", "eom"]))
        
        left = left.with_columns([
            pl.col("ret_exc_lead1m_vw_cap")
            .shift(t)
            .over("characteristics")
            .alias(f"ret_m_{t-1}")
            for t in range(1, 61)
        ])

        left = left.sort(["characteristics", "eom"])

        left = left.with_columns(
            (
                ((1 + pl.col("ret_exc_lead1m_vw_cap").shift(t*12 + 1)).log())
                .rolling_sum(window_size=12)
                .over("characteristics").exp() - 1

            ).alias(f"ret_y_{t}")
            for t in range(6, 21)
        )

        left = left.sort(["characteristics", "eom"])

        left = left.with_columns(
            pl.col("ret_exc_lead1m_vw_cap")
            .shift(1)
            .rolling_std(window_size=12*t)
            .over("characteristics")
            .alias(f"vol_m_{t}")
            for t in [2, 3, 4, 5]
        )

        left_market = left.join(market_returns, on="eom", how="left")

        left_market = left_market.sort(["characteristics", "eom"])

        left_market = left_market.with_columns([
            (
            pl.when(
                    pl.col("market_ret_exc_vw_cap")
                    .rolling_var(window_size=t)
                    .over("characteristics") > 0
                    )
                .then(
                    (
                        ((pl.col("ret_m_0") * pl.col("market_ret_exc_vw_cap"))
                        .rolling_mean(window_size=t)
                        .over("characteristics"))
                         -
                        (pl.col("ret_m_0").rolling_mean(window_size=t).over("characteristics") *
                        pl.col("market_ret_exc_vw_cap").rolling_mean(window_size=t).over("characteristics"))
                    )
                    /
                    pl.col("market_ret_exc_vw_cap")
                    .rolling_var(window_size=t)
                    .over("characteristics"))
                .otherwise(None)
            ).alias(f"beta_{t}m")
            for t in [12, 24, 36, 48, 60]
        ])

        left_market = (left_market.with_columns(
            pl.col("ret_m_0").cum_count().over("characteristics").alias("n"))
            .sort(["characteristics", "eom"]))
        
        left_market = left_market.with_columns(
            (
                ((
                    (pl.col("market_ret_exc_vw_cap") * pl.col("ret_m_0"))
                    .cum_sum().over("characteristics") / pl.col("n")
                )
                - 
                (
                    (pl.col("market_ret_exc_vw_cap").cum_sum().over("characteristics") / pl.col("n") )
                    * (pl.col("ret_m_0").cum_sum().over("characteristics") / pl.col("n"))
                ))

                /
                (
                    (pl.col("market_ret_exc_vw_cap") ** 2).cum_sum().over("characteristics") / pl.col("n") 
                    - (pl.col("market_ret_exc_vw_cap").cum_sum().over("characteristics") / pl.col("n") ) ** 2
                )
            ).alias("beta_full")
        ).drop("n")

        right = lms_returns.select(["eom", "characteristics", "chr_spread"])
        right_pivot = right.pivot(on="characteristics", index="eom", values="chr_spread")
        right_pivot = right_pivot.rename(lambda c: f"{c}_spread" if c != "eom" else c)

        all = left_market.join(right_pivot, on="eom", how="left")

        to_clean_cols = [c for c in all.columns if (
            c.startswith(("ret_m_", "ret_y_", "vol_m_", "beta_")) or c.endswith("_spread"))]
        all = all.filter(
            pl.all_horizontal([pl.col(c).is_not_null() for c in to_clean_cols]))
        
        all = all.sort(["eom", "characteristics"])
        all = all.with_columns(pl.arange(0, pl.len()).alias("row_id"))

        meta = all.select(["row_id", "eom", "characteristics", "excntry"]).sort("row_id")
        X = all.drop(["ret_exc_lead1m_vw_cap", "excntry", "characteristics", "eom"]).sort("row_id")
        y = all.select(["row_id", "ret_exc_lead1m_vw_cap"]).sort("row_id")

        X.write_parquet(data_path / "factor_characteristics" / f"{excntry}_{pfs}_feature.parquet")
        y.write_parquet(data_path / "factor_characteristics" / f"{excntry}_{pfs}_target.parquet")
        meta.write_parquet(data_path / "factor_characteristics" / f"{excntry}_meta.parquet")


def build_train_val_test_idx(data_path, excntry, min_train_val, val, test_range, forward_steps):

    test = (test_range * 12)

    meta = pl.read_parquet(
        data_path/"factor_characteristics"/f"{excntry}_meta.parquet").to_pandas()
    
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

def eval_correlation(df, excntry, data_path):

    meta = pl.read_parquet(
        data_path/"factor_characteristics"/f"{excntry}_meta.parquet").to_pandas()
    
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


def predict_with_ols(X_train,
                     y_train,
                      X_test,
                      y_test,
                      r2_split,
                      y_pred_split,
                      y_true_split):

    
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"ols - r2_pred: {r2} for this split")

    r2_split.setdefault("OLS", []).append(r2)
    y_pred_split.setdefault("OLS", []).append(y_pred.ravel())
    y_true_split.setdefault("OLS", []).append(y_test.values.ravel())


def predict_with_pls(X_train,
                     y_train,
                     X_val,
                     y_val,
                     X_train_val,
                     y_train_val,
                     X_test,
                     y_test,
                     r2_split,
                     y_pred_split,
                     y_true_split):
    
    ncomp = np.arange(1, 11) 
    best_r2_val = -np.inf
    best_K_val = None

    for K in ncomp:
        pls = PLSRegression(n_components=K)
        pls.fit(X_train, y_train)
        y_val_pred = pls.predict(X_val) 

        r2_val = r2_score(y_val, y_val_pred)

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_K_val = K

    print(f"pls - best r2 val: {best_r2_val} for K:{best_K_val}")

    pls_best = PLSRegression(n_components=best_K_val)
    pls_best.fit(X_train_val, y_train_val)
    
    y_pred = pls_best.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"pls - r2_pred: {r2} for this split")

    r2_split.setdefault("PLS", []).append(r2)
    y_pred_split.setdefault("PLS", []).append(y_pred.ravel())
    y_true_split.setdefault("PLS", []).append(y_test.values.ravel())

def predict_with_lasso(X_train,
                     y_train,
                     X_val,
                     y_val,
                     X_train_val,
                     y_train_val,
                     X_test,
                     y_test,
                     r2_split,
                     y_pred_split,
                     y_true_split):

    #alpha_values = [0.002]
    alpha_values = np.logspace(-4, np.log10(0.002), 100)
    best_r2_val = -np.inf
    best_alpha_val = None

    for alpha in alpha_values:

        lasso = Lasso(alpha=alpha, max_iter=100000)
        lasso.fit(X_train, y_train)
        y_pred_val = lasso.predict(X_val)
        r2_val = r2_score(y_val, y_pred_val)

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_alpha_val = alpha

    print(f"lasso - best r2_val: {best_r2_val} for alpha: {best_alpha_val}")

    lasso_best = Lasso(alpha=best_alpha_val, max_iter=100000)
    lasso_best.fit(X_train_val, y_train_val)

    n_total = len(lasso_best.coef_)
    n_zero = (lasso_best.coef_ == 0).sum()

    print(f"Zero coefficients: {n_zero}/{n_total}")

    y_pred = lasso_best.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"lasso - r2_pred: {r2} for this split")


    r2_split.setdefault("LASSO", []).append(r2)
    y_pred_split.setdefault("LASSO", []).append(y_pred.ravel())
    y_true_split.setdefault("LASSO", []).append(y_test.values.ravel())


def predict_with_ridge(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):

    alpha_values = np.logspace(-4, 4, 100)
    best_r2_val = -np.inf
    best_alpha = None

    for alpha in alpha_values:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_val = ridge.predict(X_val)
        r2_val = r2_score(y_val, y_pred_val)

        if r2_val > best_r2_val:
            best_r2_val = r2_val
            best_alpha = alpha

    print(f"ridge - best r2_val: {best_r2_val} for alpha: {best_alpha}")

    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train_val, y_train_val)

    y_pred = ridge_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"ridge - r2_pred: {r2} for this split")

    r2_split.setdefault("RIDGE", []).append(r2)
    y_pred_split.setdefault("RIDGE", []).append(y_pred.ravel())
    y_true_split.setdefault("RIDGE", []).append(y_test.values.ravel())

def predict_with_enet(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):
    
    alpha_values = np.logspace(-4, 0, 100)
    #alpha_values = np.logspace(-4, np.log10(0.003), 100)
    #alpha_values = [0.002]
    best_r2_val = -np.inf
    best_param = None

    for alpha in alpha_values:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:

            enet = ElasticNet(alpha=alpha, l1_ratio= l1, max_iter=100000)
            enet.fit(X_train, y_train)
            y_pred_val = enet.predict(X_val)
            r2_val = r2_score(y_val, y_pred_val)

            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_param = (alpha, l1)

    print(f"enet - best r2_val: {best_r2_val} for alpha: {best_param}")
    best_alpha, best_l1 = best_param
    enet_best = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=100000)
    enet_best.fit(X_train_val, y_train_val)

    n_total = len(enet_best.coef_)
    n_zero = (enet_best.coef_ == 0).sum()

    print(f"Zero coefficients: {n_zero}/{n_total}")

    y_pred = enet_best.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"enet - r2_pred: {r2} for this split")

    r2_split.setdefault("ENET", []).append(r2)
    y_pred_split.setdefault("ENET", []).append(y_pred.ravel())
    y_true_split.setdefault("ENET", []).append(y_test.values.ravel())


def predict_with_rf(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):

    best_r2_val = -np.inf
    best_param = None

    for n_est in [100, 150, 200, 300]:
        for max_d in [2, 3, 4, 5]:
            for min_l in [5, 10, 20]:
                for max_f in ["sqrt", 0.3, 0.5]:

                    rf = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_leaf=min_l,
                        max_features=max_f,
                        random_state=42)
                    
                    rf.fit(X_train, y_train)
                    y_pred_val = rf.predict(X_val)
                    r2_val = r2_score(y_val, y_pred_val)
                    
                    if r2_val > best_r2_val:
                        best_r2_val = r2_val
                        best_param = (n_est, max_d, min_l, max_f)

    print(f"rf - best r2_val: {best_r2_val} for parameters: {best_param}")

    best_n_est, best_max_d, best_min_l, best_max_f = best_param
    rf_best = RandomForestRegressor(
        n_estimators=best_n_est,
        max_depth=best_max_d,
        min_samples_leaf=best_min_l,
        max_features=best_max_f,
        random_state=42)
    
    rf_best.fit(X_train_val, y_train_val)
    y_pred = rf_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"rf - r2_pred: {r2} for this split")

    r2_split.setdefault("RF", []).append(r2)
    y_pred_split.setdefault("RF", []).append(y_pred.ravel())
    y_true_split.setdefault("RF", []).append(y_test.values.ravel())

def predict_with_gbrt(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):

    best_r2_val = -np.inf
    best_param = None

    for lr in [0.01, 0.02, 0.05]:
        for n_est in [100, 150, 200, 300, 400, 600]:
            for max_d in [2, 3, 5, 7]:
                for min_l in [5, 10, 20]:
                    for max_f in [0.3, 0.5, "sqrt"]:
                        
                        gbrt = GradientBoostingRegressor(
                            learning_rate= lr,
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_leaf=min_l,
                            max_features=max_f,
                            random_state=42)
                        
                        gbrt.fit(X_train, y_train)
                        y_pred_val = gbrt.predict(X_val)
                        r2_val = r2_score(y_val, y_pred_val)
                        
                        if r2_val > best_r2_val:
                            best_r2_val = r2_val
                            best_param = (lr, n_est, max_d, min_l, max_f)

    print(f"gbrt - best r2_val: {best_r2_val} for parameters: {best_param}")

    best_lr, best_n_est, best_max_d, best_min_l, best_max_f = best_param

    gbrt_best = GradientBoostingRegressor(
        learning_rate=best_lr,
        n_estimators=best_n_est,
        max_depth=best_max_d,
        min_samples_leaf=best_min_l,
        max_features=best_max_f,
        random_state=42)
    
    gbrt_best.fit(X_train_val, y_train_val)
    y_pred = gbrt_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"gbrt - r2_pred: {r2} for this split")

    r2_split.setdefault("GBRT", []).append(r2)
    y_pred_split.setdefault("GBRT", []).append(y_pred.ravel())
    y_true_split.setdefault("GBRT", []).append(y_test.values.ravel())

def predict_with_xgb(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):

    best_r2_val = -np.inf
    best_param = None

    for lr in [0.01, 0.03, 0.05]:
        for n_est in [200, 300, 500]:
            for max_d in [3, 5, 7]:
                for subs in [0.6, 0.8, 1.0]:
                    for colsample in [0.6, 0.8, 1.0]:

                        xgb = XGBRegressor(
                            learning_rate=lr,
                            n_estimators=n_est,
                            max_depth=max_d,
                            subsample=subs,
                            colsample_bytree=colsample,
                            reg_lambda=1.0,
                            reg_alpha=0.0,
                            random_state=42,
                            n_jobs=-1
                        )

                        xgb.fit(X_train, y_train)
                        y_pred_val = xgb.predict(X_val)
                        r2_val = r2_score(y_val, y_pred_val)

                        if r2_val > best_r2_val:
                            best_r2_val = r2_val
                            best_param = (lr, n_est, max_d, subs, colsample)

    print(f"xgb - best r2_val: {best_r2_val} for parameters: {best_param}")

    best_lr, best_n_est, best_max_d, best_subs, best_colsample = best_param

    xgb_best = XGBRegressor(
        learning_rate=best_lr,
        n_estimators=best_n_est,
        max_depth=best_max_d,
        subsample=best_subs,
        colsample_bytree=best_colsample,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1
    )

    xgb_best.fit(X_train_val, y_train_val)

    y_pred = xgb_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"xgb - r2_pred: {r2} for this split")

    r2_split.setdefault("XGB", []).append(r2)
    y_pred_split.setdefault("XGB", []).append(y_pred.ravel())
    y_true_split.setdefault("XGB", []).append(y_test.values.ravel())

def predict_with_ffnn(
        X_train, y_train,
        X_val, y_val,
        X_train_val, y_train_val,
        X_test, y_test,
        r2_split, y_pred_split, y_true_split):
    
    best_r2_val = -np.inf
    best_param = None

    for lr in [1e-4, 5e-4, 1e-3]: 
        for alpha in [1e-5, 1e-4, 1e-3, 1e-2]: 
            ffnn = MLPRegressor(
                hidden_layer_sizes=(8,),   
                activation="relu",        
                solver="adam",            
                learning_rate_init=lr,
                alpha=alpha,               
                max_iter=500,
                random_state=42,
                early_stopping=True)
            
            ffnn.fit(X_train, y_train)
            y_pred_val = ffnn.predict(X_val)
            r2_val = r2_score(y_val, y_pred_val)
            
            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_param = (lr, alpha)

    print(f"ffnn - best r2_val: {best_r2_val} for parameters: {best_param}")

    best_lr, best_alpha = best_param
    
    ffnn_best = MLPRegressor(
                hidden_layer_sizes=(8,),   
                activation="relu",        
                solver="adam",            
                learning_rate_init=best_lr,
                alpha=best_alpha,               
                max_iter=500,
                random_state=42,
                early_stopping=True)
    
    ffnn_best.fit(X_train_val, y_train_val)
    y_pred = ffnn_best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"ffnn - r2_pred: {r2} for this split")

    r2_split.setdefault("FFNN", []).append(r2)
    y_pred_split.setdefault("FFNN", []).append(y_pred.ravel())
    y_true_split.setdefault("FFNN", []).append(y_test.values.ravel())

def predict_with_comb(y_pred_split, y_true_split, r2_split, base_models):

    base_models = [m for m in base_models if m != "COMB"]
    lengths = [len(y_pred_split[m]) for m in base_models]
    assert len(set(lengths)) == 1, "mismatch in splits across models"

    y_pred_split_latest = [y_pred_split[m][-1] for m in base_models]
    
    y_pred_comb = np.vstack(y_pred_split_latest)
    y_pred = y_pred_comb.mean(axis=0)
    y_test = y_true_split[base_models[0]][-1]
    
    r2 = r2_score(y_test, y_pred)
    print(f"comb - r2_pred: {r2} for this split")

    r2_split.setdefault("COMB", []).append(r2)
    y_pred_split.setdefault("COMB", []).append(y_pred.ravel())
    y_true_split.setdefault("COMB", []).append(y_test.ravel())

def train_pred_model(data_path, excntry, pfs, splits_idx, model = "all"):
    
    X = pl.read_parquet(
            data_path/"factor_characteristics"/f"{excntry}_{pfs}_feature.parquet"
            ).to_pandas().drop(columns="row_id")
    
    y = pl.read_parquet(
            data_path/"factor_characteristics"/f"{excntry}_{pfs}_target.parquet"
            ).to_pandas()
    
    y_series = y["ret_exc_lead1m_vw_cap"]

    r2_split = {}
    y_pred_split = {}
    y_true_split = {}
    test_idx_split = []

    if model == "all":
        models_to_run = ["OLS", "PLS", "LASSO", "ENET", "RF", "GBRT", "FFNN", "COMB"]
    elif isinstance(model, str):
        models_to_run = [model.upper()]
    else:
        models_to_run = [m.upper() for m in model]

    for split in splits_idx:

        train_index = split["train"]
        val_index = split["val"]
        test_index = split["test"]
        train_val_index = np.concatenate([train_index, val_index])

        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]
        X_train_val = X.iloc[train_val_index]
        X_test = X.iloc[test_index]
        
        scaler_cv = StandardScaler()
        X_train = scaler_cv.fit_transform(X_train)
        X_val = scaler_cv.transform(X_val)

        scaler_best = StandardScaler()
        X_train_val = scaler_best.fit_transform(X_train_val)
        X_test = scaler_best.transform(X_test)

        y_train = y_series.iloc[train_index]
        y_val = y_series.iloc[val_index]
        y_train_val = y_series.iloc[train_val_index]
        y_test = y_series.iloc[test_index]

        test_idx_split.append(test_index)
        
        if "OLS" in models_to_run:
            predict_with_ols(
                X_train=X_train_val, y_train=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
            
        if "PLS" in models_to_run:
            predict_with_pls(
                X_train=X_train,y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val,y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
            
        if "LASSO" in models_to_run:
            predict_with_lasso(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
            
        if "RIDGE" in models_to_run:
            predict_with_ridge(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_train_val=X_train_val, y_train_val=y_train_val,
            X_test=X_test, y_test=y_test,
            r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)

        if "ENET" in models_to_run:
            predict_with_enet(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
        
        if "RF" in models_to_run:
            predict_with_rf(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
        
        if "GBRT" in models_to_run:
            predict_with_gbrt(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
        
        if "FFNN" in models_to_run:
            predict_with_ffnn( 
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
            
        if "XGB" in models_to_run:
            predict_with_xgb(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_train_val=X_train_val, y_train_val=y_train_val,
                X_test=X_test, y_test=y_test,
                r2_split=r2_split, y_pred_split=y_pred_split, y_true_split=y_true_split)
            
        if "COMB" in models_to_run:
            predict_with_comb(
                y_pred_split=y_pred_split,
                y_true_split=y_true_split,
                r2_split=r2_split,
                base_models=models_to_run)


    return r2_split, y_pred_split, y_true_split, test_idx_split

def eval_model(
        data_path, pfs, r2_split, y_pred_split, y_true_split, test_idx_split, excntry):
    
    results = []
    all_models_out = []

    test_idx_all = np.concatenate(test_idx_split)

    for model in y_pred_split.keys():

        y_pred_all = np.concatenate(y_pred_split[model])
        y_true_all = np.concatenate(y_true_split[model])

        r2_all = r2_score(y_true_all, y_pred_all)
        
        model_out = pd.DataFrame({
            "row_id": test_idx_all,
            "prediction": y_pred_all,
            "actual": y_true_all,
            "model": model,
            "excntry":excntry,
            "pfs": pfs}
        )
        
        out_corr = eval_correlation(df=model_out, excntry=excntry, data_path=data_path)
        
        results.append({
        "country": excntry,
        "pfs": pfs,
        "model": model,
        "y_pred": y_pred_all,
        "y_true": y_true_all,
        "test_idx": test_idx_all,
        "r2": r2_all,
        "r2_split": r2_split.get(model, []),
        "spearman": out_corr["spearman"],
        "pearson": out_corr["pearson"]})

        out_path = data_path/"ml_model_output"/f"{excntry}_{pfs}_ml_models.parquet"
        model_out.to_parquet(out_path, index=False)
        
    return results


def build_strategy_returns(
        data_path, model_path, meta_path, excntry, pfs, n_buckets):

        monthly_lst = []
        global_lst = []
        bucket_lst = []

        ml_out = pl.read_parquet(model_path)
        meta = pl.read_parquet(meta_path)
        eom_pred = ml_out.join(meta["row_id", "eom"], on="row_id", how="left")

        models = eom_pred.select("model").unique().to_series().to_list()

        for model in models:

            ml_buckets = (
                eom_pred
                .filter(pl.col("model") == model)
                .with_columns(
                    pl.col("prediction")
                    .rank(method="ordinal").over("eom").alias("rank"))
                .with_columns(
                    ((
                        (pl.col("rank") - 1) / pl.len().over("eom")*n_buckets
                        ).floor().cast(pl.Int64) + 1
                        ).alias("bucket"))
                )
            
            port_ret_monthly = (
                ml_buckets
                .group_by("eom", "bucket")
                .agg(pl.col("actual").mean().alias("mean_ret_bucket_monthly"))
                .with_columns(pl.lit(model).alias("model"))
                .with_columns(pl.lit(excntry).alias("excntry"))
                .with_columns(pl.lit(pfs).alias("pfs"))
                )
            
            port_ret_monthly = (
                port_ret_monthly
                .with_columns(
                    (pl.col("mean_ret_bucket_monthly")
                    .filter(pl.col("bucket") == n_buckets)
                    .first()
                    .over("eom")
                    -
                    pl.col("mean_ret_bucket_monthly")
                    .filter(pl.col("bucket") == 1)
                    .first()
                    .over("eom")
                ).alias("hml_ret_monthly")
            ))

            port_ret_global = (
                ml_buckets
                .group_by("bucket")
                .agg(pl.col("actual").mean().alias("mean_ret_bucket_global"))
                .with_columns(pl.lit(model).alias("model"))
                .with_columns(pl.lit(excntry).alias("excntry"))
                .with_columns(pl.lit(pfs).alias("pfs"))
            )

            port_ret_global = (
                port_ret_global
                .with_columns(
                    (pl.col("mean_ret_bucket_global")
                    .filter(pl.col("bucket") == n_buckets)
                    .first()
                    -
                    pl.col("mean_ret_bucket_global")
                    .filter(pl.col("bucket") == 1)
                    .first()
                ).alias("hml_ret_global")
    
            ))
            
            bucket_lst.append(ml_buckets)
            monthly_lst.append(port_ret_monthly)
            global_lst.append(port_ret_global)

        
        all_port_ret_bucket = pl.concat(bucket_lst)
        out_path_buck = data_path/"portfolio_returns"/f"{excntry}_{pfs}_port_ret_bucket.parquet"
        all_port_ret_bucket.write_parquet(out_path_buck)

        all_port_ret_monthly = pl.concat(monthly_lst)
        out_path_mon = data_path/"portfolio_returns"/f"{excntry}_{pfs}_port_ret_monthly_avg.parquet"
        all_port_ret_monthly.write_parquet(out_path_mon)

        all_port_ret_global = pl.concat(global_lst)
        out_path_gl = data_path/"portfolio_returns"/f"{excntry}_{pfs}_port_ret_global_avg.parquet"
        all_port_ret_global.write_parquet(out_path_gl)







