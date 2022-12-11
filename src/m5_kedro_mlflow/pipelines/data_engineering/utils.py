import pandas as pd


# utils
# utils
def create_ma_and_ma_diff(df, MA: list, key: list, lag: int) -> pd.DataFrame:
    key_name = ("_".join(key) if len(key) > 0 else "global") + "_"

    # create lag for ma
    df_temp = (
        df[key + ["sold", "date"]]
        .groupby(key + ["date"])
        .sum()
        .shift(lag)
        .reset_index()
        .rename(columns={"sold": f"{key_name}sold_lag{lag}"})
    )

    # create ma
    for ma in MA:
        if len(key) == 0:
            # create ma
            df_temp[f"{key_name}sold_lag{lag}_ma{ma}"] = df_temp[
                [f"{key_name}sold_lag{lag}"]
            ].transform(lambda x: x.rolling(ma).mean())
            # create ma diff
            df_temp[f"{key_name}sold_lag{lag}_ma{ma}_diff"] = df_temp[
                [f"{key_name}sold_lag{lag}_ma{ma}"]
            ].diff(1)
        else:
            # create ma
            df_temp[f"{key_name}sold_lag{lag}_ma{ma}"] = (
                df_temp[key + [f"{key_name}sold_lag{lag}"]]
                .groupby(key)[f"{key_name}sold_lag{lag}"]
                .transform(lambda x: x.rolling(ma).mean())
            )
            # create ma diff
            df_temp[f"{key_name}sold_lag{lag}_ma{ma}_diff"] = (
                df_temp[key + [f"{key_name}sold_lag{lag}_ma{ma}"]]
                .groupby(key)[f"{key_name}sold_lag{lag}_ma{ma}"]
                .diff(1)
            )

    df_temp = df_temp[
        [col for col in df_temp.columns if col != f"{key_name}sold_lag{lag}"]
    ]
    df = df.merge(df_temp, how="left", on=key + ["date"])
    return df
