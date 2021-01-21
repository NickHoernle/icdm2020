import os
import torch
import json

import pandas as pd
import numpy as np

from tqdm import tqdm


def get_activity_helper(posts_df, user_id):
    try:
        result = posts_df.loc[user_id]
    except:
        result = pd.Series(data=np.zeros(posts_df.shape[1]).astype(int), index=posts_df.columns)
    return result


def load_and_transform_data(in_data_path: str, threshold_achievement: int):

    posts_df = pd.read_pickle(os.path.join(in_data_path, "posts_df.pkl.gz"), compression="gzip")
    reps_df = pd.read_pickle(os.path.join(in_data_path, "reputation_df.pkl.gz"), compression="gzip")

    for col in posts_df.columns:
        posts_df[col] = posts_df[col].astype(int)

    for col in ['UserId', 'RepDayId', 'Count', 'Sum']:
        if col == 'Sum':
            reps_df.Sum.fillna(0, inplace=True)
        reps_df[col] = reps_df[col].astype(int)

    ta = threshold_achievement

    day_or_week = "Day"
    edits = reps_df[reps_df.RepText == 'edit']
    edits = edits[['UserId', 'RepDayId', 'PostTypeId', 'Count']].rename(columns={"RepDayId": "PostDayId"})
    edits["PostTypeId"] = 3

    posts_df = pd.concat([posts_df, edits], axis=0)

    p_dfs = []
    for post_type in [1, 2, 3]:
        p_dfs.append(posts_df[posts_df.PostTypeId == post_type]
            .groupby(['UserId', f'Post{day_or_week}Id']).Count.sum().reset_index()
            .pivot(index='UserId', columns=f'Post{day_or_week}Id', values='Count')
            .fillna(0))

    r_df1 = reps_df.groupby(["UserId", f"Rep{day_or_week}Id"]) \
        .Sum.sum() \
        .groupby("UserId") \
        .cumsum() \
        .reset_index()

    r_df = r_df1 \
        .loc[(r_df1.Sum >= 1000)] \
        .groupby('UserId').head(1)

    r_df2 = r_df1.pivot(index='UserId', columns=f'Rep{day_or_week}Id', values='Sum').ffill(axis=1).bfill(axis=1)

    # fill in coluns if they don't exist.
    min_col = r_df2.columns.min()
    max_col = r_df2.columns.max()
    col_names = list(range(min_col, max_col+1))


    print("Adding missing columns")
    for col in range(min_col, max_col+1):
        for p_df in p_dfs:
            if col not in p_df.columns:
                p_df[col] = 0

        if col not in r_df2.columns:
            r_df2[col] = r_df2[col - 1]

    p_dfs = [p_df[col_names] for p_df in p_dfs]

    for p_df in p_dfs:
        print(p_df.shape)
    print(r_df2.shape)

    user_ids = []
    answer_counts = []
    question_counts = []
    edit_counts = []
    reputations = []

    print("Joining the dataframes")
    joined = r_df.merge(r_df2, how="inner", on="UserId", suffixes=(None, None))

    for user, row in tqdm(joined.iterrows(), total=len(joined), desc='activity at threshold'):

        week_of_crossing = int(row[f'Rep{day_or_week}Id'])

        if week_of_crossing-ta-min_col <= 0:
            continue
        if week_of_crossing+ta >= max_col - 1:
            continue

        user_id = int(row['UserId'])

        my_reputation = row[col_names]

        if my_reputation.loc[week_of_crossing + ta+1] > 5000:
            continue
        if my_reputation.loc[week_of_crossing + ta+1] < 1010:
            continue

        my_questions = get_activity_helper(p_dfs[0], user_id)
        my_answers = get_activity_helper(p_dfs[1], user_id)
        my_edits = get_activity_helper(p_dfs[2], user_id)

        reputations.append(my_reputation.loc[week_of_crossing-ta:week_of_crossing+ta].values.astype(int))
        question_counts.append(my_questions.loc[week_of_crossing-ta:week_of_crossing+ta].values.astype(int))
        answer_counts.append(my_answers.loc[week_of_crossing - ta:week_of_crossing + ta].values.astype(int))
        edit_counts.append(my_edits.loc[week_of_crossing - ta:week_of_crossing + ta].values.astype(int))

        user_ids.append(user_id)

    reputations_data = np.array(reputations)
    question_data = np.array(question_counts)
    answer_data = np.array(answer_counts)
    edit_data = np.array(edit_counts)

    user_ids = np.array(user_ids)

    return user_ids, (reputations_data, question_data, answer_data, edit_data)


def create_reputation_dataset(
        in_data_path: str = 'data',
        out_data_path: str = 'data/reputation_data',
        threshold_achievement: int = 100
):
    user_ids, activity_data = load_and_transform_data(in_data_path, threshold_achievement)
    print(f"Total of {len(user_ids)} user trajectories")

    for user_id, reputation, question, answer, edits in tqdm(
                                zip(user_ids, *activity_data),
                            total=len(user_ids),
                            desc='dumping activity data'):
        activity_data = np.stack((question, answer, edits, reputation), axis=0)
        torch.save(activity_data, os.path.join(out_data_path, f'user_{user_id}.pt'))

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
