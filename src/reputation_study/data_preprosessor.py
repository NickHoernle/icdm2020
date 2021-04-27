import os
import torch
import json

import pandas as pd
import numpy as np

from tqdm import tqdm


############################################
# Reputation Data
############################################

def get_activity_helper(posts_df, user_id):
    try:
        result = posts_df.loc[user_id]
    except:
        result = pd.Series(data=np.zeros(posts_df.shape[1]).astype(int), index=posts_df.columns)
    return result


def load_data(in_data_path: str):
    posts_df = pd.read_pickle(os.path.join(in_data_path, "posts_df.pkl.gz"), compression="gzip")
    reps_df = pd.read_pickle(os.path.join(in_data_path, "reputation_df.pkl.gz"), compression="gzip")

    for col in posts_df.columns:
        posts_df[col] = posts_df[col].astype(int)

    for col in ['UserId', 'RepTimeId', 'Count', 'Sum']:
        if col == 'Sum':
            reps_df.Sum.fillna(0, inplace=True)
        reps_df[col] = reps_df[col].astype(int)

    return posts_df, reps_df


def add_edits(posts_df, reps_df):
    edits = reps_df[reps_df.RepText == 'edit']
    edits = edits[['UserId', 'RepTimeId', 'PostTypeId', 'Count']].rename(columns={"RepTimeId": "PostTimeId"})
    edits["PostTypeId"] = 3

    posts_df = pd.concat([posts_df, edits], axis=0)
    return posts_df


def rename_post_time(posts_df, reps_df):
    posts_df.rename(columns={"PostTimeId": "TimeId"}, inplace=True)
    reps_df.rename(columns={"RepTimeId": "TimeId"}, inplace=True)
    reps_df.drop(columns=["Count", "RepText"], inplace=True)

    posts_df = posts_df.astype('int32')
    reps_df.dropna(inplace=True)
    reps_df = reps_df.astype('int32')

    return posts_df, reps_df


def drop_unwanted_users(posts_df, reps_df, target_reputation=2000):
    users_to_drop = reps_df.groupby("UserId").Sum.sum() < target_reputation
    users_to_drop = users_to_drop[users_to_drop].index

    print(f"Dropping {len(users_to_drop)} users below threshold")

    reps_df = reps_df[~reps_df.UserId.isin(users_to_drop)]
    posts_df = posts_df[posts_df.UserId.isin(reps_df.UserId.unique())]

    return posts_df, reps_df


def preprocess_dfs(posts_df, reps_df, target_reputation=2000):
    p_dfs = []
    for post_type in [1, 2, 3]:
        p_dfs.append(posts_df[posts_df.PostTypeId == post_type]
                     .groupby(['UserId', f'TimeId']).Count.sum().reset_index()
                     .pivot(index='UserId', columns=f'TimeId', values='Count')
                     .fillna(0))

    del posts_df

    r_df1 = reps_df.groupby(["UserId", f"TimeId"]) \
        .Sum.sum() \
        .groupby("UserId") \
        .cumsum() \
        .reset_index()

    del reps_df

    r_df = r_df1 \
        .loc[(r_df1.Sum >= target_reputation)] \
        .groupby('UserId').head(1)

    r_df2 = r_df1.pivot(index='UserId', columns=f'TimeId', values='Sum').ffill(axis=1).bfill(axis=1)

    # fill in coluns if they don't exist.
    min_col = r_df2.columns.min()
    max_col = r_df2.columns.max()
    col_names = list(range(min_col, max_col + 1))

    print("Adding missing columns")
    for col in range(min_col, max_col + 1):
        for p_df in p_dfs:
            if col not in p_df.columns:
                p_df[col] = 0

        if col not in r_df2.columns:
            r_df2[col] = r_df2[col - 1]

    p_dfs = [p_df[col_names] for p_df in p_dfs]

    for p_df in p_dfs:
        print(p_df.shape)
    print(r_df2.shape)

    return p_dfs, r_df2, r_df


def load_and_transform_by_reputation(in_data_path: str, out_data_path: str, threshold_achievement: int):
    ta = threshold_achievement
    target_reputation = 2000
    total_cutoff = 3000
    reputation_range = 500, 3000

    posts_df, reps_df = load_data(in_data_path)
    posts_df = add_edits(posts_df, reps_df)
    posts_df, reps_df = rename_post_time(posts_df, reps_df)
    posts_df, reps_df = drop_unwanted_users(posts_df, reps_df, target_reputation=target_reputation)
    print("Reshaping the data")
    p_dfs, r_df2, r_df = preprocess_dfs(posts_df, reps_df, target_reputation=target_reputation)

    min_col = r_df2.columns.min()
    max_col = r_df2.columns.max()
    col_names = list(range(min_col, max_col + 1))

    joined = r_df.merge(r_df2, how="inner", on="UserId", suffixes=(None, None))

    base = pd.DataFrame({"reputation": np.arange(reputation_range[0], reputation_range[1])})

    user_ids = []
    counter = 0
    for user, row in tqdm(joined.iterrows(), total=len(joined), desc='activity at threshold'):
        date_of_crossing = int(row[f'TimeId'])
        if date_of_crossing-ta-min_col <= 0:
            continue
        if date_of_crossing+ta >= max_col - 1:
            continue

        user_id = int(row['UserId'])
        my_reputation = row[col_names]

        if my_reputation.loc[date_of_crossing + ta + 1] > reputation_range[1]:
            continue
        if my_reputation.loc[date_of_crossing + ta + 1] < target_reputation + 5:
            continue

        my_questions = get_activity_helper(p_dfs[0], user_id)
        my_answers = get_activity_helper(p_dfs[1], user_id)
        my_edits = get_activity_helper(p_dfs[2], user_id)

        my_activity = {
            "questions": my_questions.loc[date_of_crossing - ta:date_of_crossing + ta].values.astype(int),
            "answers": my_answers.loc[date_of_crossing - ta:date_of_crossing + ta].values.astype(int),
            "edits": my_edits.loc[date_of_crossing - ta:date_of_crossing + ta].values.astype(int),
            "reputation": my_reputation.loc[date_of_crossing - ta:date_of_crossing + ta].values.astype(int),
        }
        data = pd.DataFrame(my_activity).groupby("reputation").sum().reset_index().merge(base, on="reputation", how="right").fillna(0)
        data = data[(data.reputation >= reputation_range[0]) & (data.reputation < reputation_range[1])]
        data = data.groupby(data.reputation // 10).agg("sum")[["questions", "answers", "edits"]].reset_index()

        user_ids.append(user_id)
        activity_data = torch.tensor(data.values.astype(int).T)
        torch.save(activity_data, os.path.join(out_data_path, f'user_{user_id}.pt'))
        counter += 1
        if counter>20000:
            break
    print(data.columns)

    user_ids = np.array(user_ids)
    return user_ids



def load_and_transform_data(
        in_data_path: str,
        out_data_path: str,
        threshold_achievement: int,
        target_reputation: int,
        reputation_range: list
):
    ta = threshold_achievement
    reputation_range = reputation_range[0], reputation_range[1]

    posts_df, reps_df = load_data(in_data_path)
    posts_df = add_edits(posts_df, reps_df)
    posts_df, reps_df = rename_post_time(posts_df, reps_df)
    posts_df, reps_df = drop_unwanted_users(posts_df, reps_df, target_reputation=target_reputation)
    print("Reshaping the data")
    p_dfs, r_df2, r_df = preprocess_dfs(posts_df, reps_df, target_reputation=target_reputation)

    min_col = r_df2.columns.min()
    max_col = r_df2.columns.max()
    col_names = list(range(min_col, max_col + 1))

    joined = r_df.merge(r_df2, how="inner", on="UserId", suffixes=(None, None))

    user_ids = []
    counter = 0

    for user, row in tqdm(joined.iterrows(), total=len(joined), desc='activity at threshold'):

        week_of_crossing = int(row[f'TimeId'])

        if week_of_crossing-ta-min_col <= 0:
            continue
        if week_of_crossing+ta >= max_col - 1:
            continue

        user_id = int(row['UserId'])

        my_reputation = row[col_names]

        if my_reputation.loc[week_of_crossing + ta+1] > reputation_range[1]: # done a lot after cross
            continue
        if my_reputation.loc[week_of_crossing - ta - 1] < reputation_range[0]: # done nothing after cross
            continue

        my_questions = get_activity_helper(p_dfs[0], user_id)
        my_answers = get_activity_helper(p_dfs[1], user_id)
        my_edits = get_activity_helper(p_dfs[2], user_id)

        my_activity = {
            "questions": my_questions.loc[week_of_crossing-ta:week_of_crossing+ta].values.astype(int),
            "answers": my_answers.loc[week_of_crossing - ta:week_of_crossing + ta].values.astype(int),
            "edits": my_edits.loc[week_of_crossing - ta:week_of_crossing + ta].values.astype(int),
            "reputation": my_reputation.loc[week_of_crossing-ta:week_of_crossing+ta].values.astype(int)
        }

        data = pd.DataFrame(my_activity)

        user_ids.append(user_id)
        activity_data = torch.tensor(data.values.astype(int).T)
        torch.save(activity_data, os.path.join(out_data_path, f'user_{user_id}.pt'))
        counter += 1

    print(data.columns)

    user_ids = np.array(user_ids)

    return user_ids


def dump_user_ids(user_ids, out_data_path):
    np.random.seed(11)
    size_data = len(user_ids)

    train = np.random.choice(user_ids, size=int(np.floor(0.6 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, train)]
    validate = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)
    user_ids = user_ids[~np.in1d(user_ids, validate)]
    test = np.random.choice(user_ids, size=int(np.floor(0.2 * size_data)), replace=False)

    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)

    with open(os.path.join(out_data_path, 'data_indexes.json'), 'w') as f:
        obj = dict()
        obj['train'] = [int(u) for u in train]
        obj['test'] = [int(u) for u in test]
        obj['validate'] = [int(u) for u in validate]

        json.dump(obj, f)


############################################
# Electorate Data
############################################

def load_and_transform_electorate_data(in_data_path, threshold_achievement, badge="Electorate", not_badge=""):
    action_names = ['Answers', 'Questions', 'Comments', 'Edits', 'AnswerVotes', 'QuestionVotes', 'ReviewTasks']

    activity_df = pd.read_csv(os.path.join(in_data_path, "so_badges.csv.gz"), compression="gzip")

    activity_df.Date = pd.to_datetime(activity_df.Date)
    activity_df['time_ix'] = (activity_df.Date - pd.datetime(year=2017, month=1, day=1)).dt.days

    badge_wins = activity_df.dropna(subset=[badge]).groupby('DummyUserId')['time_ix'].first().reset_index()
    print(f"Total {badge_wins.shape} users found")
    if not_badge != "":
        not_badge = activity_df.dropna(subset=[not_badge]).groupby('DummyUserId')['time_ix'].first().reset_index()
        print(f"Dropping {not_badge.shape[0]} users")
        badge_wins = badge_wins[~badge_wins.DummyUserId.isin(not_badge.DummyUserId)]

    badge_wins.rename(columns={'time_ix': 'badge_win_date'}, inplace=True)

    df = pd.merge(activity_df[['DummyUserId', 'time_ix'] + action_names],
                  badge_wins[['DummyUserId', "badge_win_date"]], on='DummyUserId')

    user_ids = []
    activities = []

    max_time = df.time_ix.max()

    for user_id, user_actions in tqdm(df.groupby("DummyUserId"),
                                      total=df.DummyUserId.nunique(),
                                      desc='Processing user activities'):
        badge_win_date = user_actions.badge_win_date.min()
        if badge_win_date - threshold_achievement < 0:
            continue
        if max_time - badge_win_date - (threshold_achievement + 1) < 0:
            continue

        user_acts = user_actions.set_index("time_ix").loc[badge_win_date-threshold_achievement:badge_win_date+threshold_achievement]

        activities.append(user_acts[action_names].values.astype(int).T)
        user_ids.append(int(user_id))

    return np.array(user_ids), np.stack(activities)

############################################
# Executables
############################################

def create_reputation_dataset(
        in_data_path: str = 'data',
        out_data_path: str = 'data/reputation_data',
        threshold_achievement: int = 70,
        target_reputation: int = 500,
        reputation_range_u: int = 1000,
        reputation_range_l: int = 250
):
    if not os.path.exists(out_data_path):
        os.mkdir(out_data_path)
    user_ids = load_and_transform_data(in_data_path, out_data_path, threshold_achievement, target_reputation, [reputation_range_l, reputation_range_u])
    # user_ids = load_and_transform_by_reputation(in_data_path, out_data_path, threshold_achievement)

    print(f"Total of {len(user_ids)} user trajectories")
    dump_user_ids(user_ids, out_data_path)


def create_activity_dataset(
    in_data_path: str, out_data_path: str, threshold_achievement: int, badge: str, not_badge: str):
    if not os.path.exists(out_data_path):
        os.mkdir(out_data_path)

    user_ids, activity_data = load_and_transform_electorate_data(
        in_data_path,
        threshold_achievement,
        badge=badge,
        not_badge=not_badge
    )
    print(f"Total of {len(user_ids)} user trajectories")

    for user_id, activities in tqdm(
                                zip(user_ids, activity_data),
                            total=len(user_ids),
                            desc='dumping activity data'):
        activity_data = torch.tensor(activities)
        torch.save(activity_data, os.path.join(out_data_path, f'user_{user_id}.pt'))

    dump_user_ids(user_ids, out_data_path)


params = {
    "Electorate": ["Electorate", ""],
    "CivicDuty": ["CivicDuty", "Electorate"],
    "CopyEditor": ["CopyEditor", ""],
    "StrunkWhite": ["StrunkWhite", "CopyEditor"],
}


def create_dataset(in_data_path: str = 'data', threshold_achievement: int = 70, type="Electorate"):
    create_activity_dataset(in_data_path,
                            f"{in_data_path}/pt_{type.lower()}",
                            threshold_achievement,
                            params[type][0],
                            params[type][1])
