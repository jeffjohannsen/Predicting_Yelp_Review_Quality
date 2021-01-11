import pandas as pd
from sqlalchemy import create_engine


def load_dataframe_from_yelp_2(query):
    """
    Connects to yelp_2 database on Postgres and
    loads a Pandas dataframe based off sql query.

    Args:
        query (string): Sql query to select data from yelp_2.

    Returns:
        Dataframe: Pandas dataframe of records
                    from sql query of yelp_2 database.
    """
    connect = "postgresql+psycopg2://postgres:password@localhost:5432/yelp_2"
    engine = create_engine(connect)
    df = pd.read_sql(query, con=engine)
    df = df.copy()
    return df


def save_dataframe_to_yelp_2(df, table):
    """
    Connects to yelp_2 database on Postgres and
    saves a Pandas dataframe to table.

    Args:
        df (Dataframe): Dataframe to save.
        table (string): Table name to save dataframe to.
    """
    connect = "postgresql+psycopg2://postgres:password@localhost:5432/yelp_2"
    engine = create_engine(connect)
    df.to_sql(table, con=engine, if_exists="fail", index=False)
    print(f"Save to {table} successful.")


def counter(x):
    if x in ["None", None, ""]:
        return 0
    else:
        y = x.split(",")
        return len(y)


def expand_checkin_data():
    """
    Expands informational value
    of original checkins table.
    """
    query = """
            SELECT business_id,
                   date
            FROM checkin
            ;
            """
    df = load_dataframe_from_yelp_2(query)
    df["checkin_count"] = df.date.apply(counter)
    df["date_list"] = [pd.to_datetime(x) for x in df.date.str.split(",")]
    df = df.drop("date", axis=1)
    df["oldest_checkin"] = [min(lst) for lst in df["date_list"]]
    df["most_recent_checkin"] = [max(lst) for lst in df["date_list"]]

    month_list = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    date_column_list = []
    date_comparison_list = []
    for year in list(range(2004, 2021)):
        for month in month_list:
            date = f"{month} {year}"
            date_column_list.append(f"checkins_before_{month}_{year}")
            datetime = pd.to_datetime(date)
            date_comparison_list.append(datetime)

    for idx, val in enumerate(date_column_list):
        df[val] = df.date_list.apply(
            lambda x: sum(1 if y < date_comparison_list[idx] else 0 for y in x)
        )
        df[f"percent_of_{val}"] = df[val] / df["checkin_count"]

    df = df.drop("date_list", axis=1)
    save_dataframe_to_yelp_2(df, "checkin_expanded")


def expand_checkin_data_2():
    """
    Creates new checkin table in Postgres
    with only checkin count, oldest checkin,
    and most recent checkin.
    """
    query = """
        SELECT business_id,
               date
        FROM checkin
        ;
        """
    df = load_dataframe_from_yelp_2(query)
    df["checkin_count"] = df.date.apply(counter)
    df["date_list"] = [pd.to_datetime(x) for x in df.date.str.split(",")]
    df["oldest_checkin"] = [min(lst) for lst in df["date_list"]]
    df["most_recent_checkin"] = [max(lst) for lst in df["date_list"]]
    df = df.drop(["date", "date_list"], axis=1)
    save_dataframe_to_yelp_2(df, "checkin_expanded_2")


if __name__ == "__main__":
    expand_checkin_data_2()
