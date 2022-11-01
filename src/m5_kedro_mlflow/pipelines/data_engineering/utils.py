import pandas as pd


def create_lag_ma(df, LAG: list, MA: list, key: list) -> pd.DataFrame:
    key_name = "_".join(key) + "_" if len(key) > 0 else ""
    for lag in LAG:
        df_temp = (
            df[key + ["sold", "date"]]
            .groupby(key + ["date"])
            .sum()
            .shift(lag)
            .reset_index()
            .rename(columns={"sold": f"{key_name}sold_lag{lag}"})
        )

    for ma in MA:
        for lag in LAG:
            if len(key) == 0:
                df_temp[f"{key_name}sold_lag{lag}_ma{ma}"] = df_temp[
                    key + [f"{key_name}sold_lag{lag}"]
                ].transform(lambda x: x.rolling(ma).mean())
            else:
                df_temp[f"{key_name}sold_lag{lag}_ma{ma}"] = (
                    df_temp[key + [f"{key_name}sold_lag{lag}"]]
                    .groupby(key)[f"{key_name}sold_lag{lag}"]
                    .transform(lambda x: x.rolling(ma).mean())
                )

    df = df.merge(df_temp, how="left", on=key + ["date"])
    return df
