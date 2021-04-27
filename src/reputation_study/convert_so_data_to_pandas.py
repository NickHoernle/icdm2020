from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import datetime


def get_spark(appName, mem, cores):
    return (
        SparkSession
        .builder
        .config('spark.driver.memory', f'{mem}g')
        .config('spark.sql.session.timeZone', 'UTC')
        .appName('count_per_user_bin')
        .master(f'local[{cores}]')
        .getOrCreate()
    )


def get_spark_dataframes(data_dir):
    spark = get_spark('gen_timeseries', mem=16, cores=8)
    users = spark.read.format('json').load(f'{data_dir}/Users.json.gz')
    posts = spark.read.format('json').load(f'{data_dir}/Posts.json.gz')
    reps = spark.read.format('json').load(f'{data_dir}/Reps.json.gz')

    return users, posts, reps


def add_timeid(D, col_in, col_out='WeekId', time_type="week", start_date=datetime.datetime(year=2008, month=7, day=27)):
    assert time_type in ["week", "day"]
    num_seconds = 604800
    if time_type == "day":
        num_seconds = 86400
    return D.withColumn(
        col_out,
        F.floor((F.col(col_in).cast('long') - F.unix_timestamp(F.lit(start_date)).cast('long') ) / num_seconds)  # seconds per week
    )


def compute_pandas_dataframes(users, posts, reps, min_rep, max_rep, time_type="week"):
    assert time_type in ["week", "day"]
    U = (
        users
            .select(
            F.col('Id').alias('UserId'),
            F.col('Reputation').alias('UserRep'),
            F.to_timestamp('CreationDate').alias('UserCreationTime'),
            F.to_timestamp('LastAccessDate').alias('UserLastAccessTime'),
        )
            .filter(f'UserId != -1 and '
                    f'UserRep >= {min_rep} and '
                    f'UserRep < {max_rep} and '
                    f'UserCreationTime > "2012-01-01"')  # user -1 is not a real user
            .orderBy('UserCreationTime')
    )
    U = add_timeid(U, 'UserCreationTime', 'UserCreationDayId', time_type=time_type)
    U = add_timeid(U, 'UserLastAccessTime', 'UserLastAccessDayId', time_type=time_type)

    P = posts.select(
        F.col('PostTypeId'),
        F.col('OwnerUserId').alias('PostUserId'),
        F.to_timestamp('CreationDate').alias('PostCreationTime'),
    )

    R = reps.select(
        F.col('PostTypeId'),
        F.col('Delta').alias('RepDelta'),
        F.col('UserId').alias('RepUserId'),
        F.col('Text').alias('RepText'),
        F.to_timestamp('Time').alias('RepTime'),
    )

    UP = add_timeid(U.join(P, on=U.UserId == P.PostUserId, how='inner'), 'PostCreationTime', 'PostTimeId', time_type=time_type)
    UR = add_timeid(U.join(R, on=U.UserId == R.RepUserId, how='inner'), 'RepTime', 'RepTimeId', time_type=time_type)

    users_df = U.toPandas()
    posts_df = UP.groupby('UserId', 'PostTimeId', 'PostTypeId').agg(
        F.count('PostUserId').alias('Count'),
    ).toPandas()
    reps_df = UR.groupby('UserId', 'RepTimeId', 'PostTypeId', 'RepText').agg(
        F.count('RepUserId').alias('Count'),
        F.sum('RepDelta').alias('Sum'),
    ).toPandas()

    return users_df, posts_df, reps_df


def get_dataframe_from_spark():
    # data_dir = "../../data/"
    threshold = 200
    data_dir = "/Volumes/Seagate Backup Plus Drive/so_data/"
    users_df, posts_df, reps_df = get_spark_dataframes(f"{data_dir}/raw")
    print("Read into Spark")
    users_df, posts_df, reps_df = compute_pandas_dataframes(users_df, posts_df, reps_df, min_rep=threshold-10, max_rep=500000, time_type="week")
    print("Computed Dataframes")
    users_df.to_pickle(f"{data_dir}/{threshold}/users_df.pkl.gz", compression="gzip")
    print("Dumped Users")
    posts_df.to_pickle(f"{data_dir}/{threshold}/posts_df.pkl.gz", compression="gzip")
    print("Dumped Posts")
    reps_df.to_pickle(f"{data_dir}/{threshold}/reputation_df.pkl.gz", compression="gzip")
    print("Dumped Reputation")
    print("<==========================> Done <======================================>")


if __name__ == "__main__":
    get_dataframe_from_spark()
