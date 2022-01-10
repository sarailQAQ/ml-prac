# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def prepare(dataset):
    # 复制
    data = dataset.copy()
    # 折扣处理
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减转换为折扣率
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: -1 if ':' not in str(x)
    else int(str(x).split(':')[0]))  # 满减最低消费
    # 距离处理
    data['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data['Weekday_received'] = data['date_received'].apply(lambda x: x.isoweekday())

    return data


def get_label(dataset):
    # 复制
    data = dataset.copy()
    # 领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    return data


def get_label_feature(label_field):
    data = label_field.copy()
    data['Date_received'] = data['Date_received'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['cnt'] = 1  # 方便特征提取
    l_feat = data.copy()

    # 用户特征
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 用户领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户是否第一次领券
    tmp = data[keys + ['Date_received']].sort_values(['Date_received'], ascending=True)
    first = tmp.drop_duplicates(keys, keep='first')
    first[prefixs + 'is_first_receive'] = 1
    l_feat = pd.merge(l_feat, first, on=keys + ['Date_received'], how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户是否最后一次领券
    last = tmp.drop_duplicates(keys, keep='last')
    last[prefixs + 'is_last_receive'] = 1
    l_feat = pd.merge(l_feat, last, on=keys + ['Date_received'], how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领取的优惠券不同折扣率种数
    pivot = pd.pivot_table(data, index=keys, values='Discount_rate', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Discount_rate': prefixs + 'received_discount_rate_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户领券距离的平均数
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_mean_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户领券距离的最大值
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.max([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_max_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户领券距离的最小值
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.min([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_min_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户领券距离的方差
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.var([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_var_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家特征
    keys = ['Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 商家被领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 领取商家优惠券的不同用户数
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_User_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 商家优惠券被领取距离的平均数
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_mean_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家优惠券被领取距离的中位数
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.median([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_median_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 商家优惠券被领取距离的方差
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.var([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_var_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 优惠券特征
    keys = ['Coupon_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 领取该优惠券的用户数
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'User_id': prefixs + 'received_user_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 领券距离的平均数
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.mean([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_mean_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 领券距离的中位数
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.median([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_median_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 领券距离的方差
    pivot = pd.pivot_table(data, index=keys, values='Distance',
                           aggfunc=lambda x: np.var([np.nan if i == -1 else i for i in x]))
    pivot = pd.DataFrame(pivot).rename(columns={'Distance': prefixs + 'received_var_distance'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(-1, downcast='infer', inplace=True)

    # 用户-商家特征
    keys = ['User_id', 'Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 该用户在该商家领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户是否第一次在该商家领取优惠券
    tmp = data[keys + ['Date_received']].sort_values(['Date_received'], ascending=True)
    first = tmp.drop_duplicates(keys, keep='first')
    first[prefixs + 'is_first_receive'] = 1
    l_feat = pd.merge(l_feat, first, on=keys + ['Date_received'], how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户是否最后一次在该商家领取优惠券
    last = tmp.drop_duplicates(keys, keep='last')
    first[prefixs + 'is_last_receive'] = 1
    l_feat = pd.merge(l_feat, first, on=keys + ['Date_received'], how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 该用户在该商家领取的优惠券种数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={'Coupon_id': prefixs + 'received_coupon_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户-优惠券特征
    keys = ['User_id', 'Coupon_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 用户领取特定优惠券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户-领取日期特征
    keys = ['User_id', 'Date_received']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 用户当天领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 商家-领取日期特征
    keys = ['Merchant_id', 'Date_received']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 商家当天被领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'recieved_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户-商家-领取日期特征
    keys = ['User_id', 'Merchant_id', 'Date_received']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 用户当天领取该商家优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户是否在同一天重复在该商家领取优惠券
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=lambda x: 1 if len(x) > 1 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'repeat_receive'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户当天领取优惠券的平均值
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_mean_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户当天领取优惠券的方差
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=np.var)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_var_cnt'}).reset_index()
    l_feat = pd.merge(l_feat, pivot, on=keys, how='left')
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 用户
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 用户-距离正反排序
    l_feat[prefixs + 'distance_true_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=True)
    l_feat[prefixs + 'distance_false_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=False)

    # 用户-领券日期正反排序
    l_feat[prefixs + 'date_received_true_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=True)
    l_feat[prefixs + 'date_received_false_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=False)

    # 用户-折扣率正反排序
    l_feat[prefixs + 'discount_rate_true_rank'] = l_feat.groupby(keys)['discount_rate'].rank(ascending=True)
    l_feat[prefixs + 'discount_rate_false_rank'] = l_feat.groupby(keys)['discount_rate'].rank(ascending=False)

    # 用户-满减最低消费正反排序
    l_feat[prefixs + 'min_cost_of_manjian_true_rank'] = l_feat.groupby(keys)['min_cost_of_manjian'].rank(ascending=True)
    l_feat[prefixs + 'min_cost_of_manjian_false_rank'] = l_feat.groupby(keys)['min_cost_of_manjian'].rank(
        ascending=False)

    # 商家
    keys = ['Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'

    # 商家-距离正反排序
    l_feat[prefixs + 'distance_true_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=True)
    l_feat[prefixs + 'distance_false_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=False)

    # 商家-领券日期正反排序
    l_feat[prefixs + 'date_received_true_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=True)
    l_feat[prefixs + 'date_received_false_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=False)

    # 商家-折扣率正反排序
    l_feat[prefixs + 'discount_rate_true_rank'] = l_feat.groupby(keys)['discount_rate'].rank(ascending=True)
    l_feat[prefixs + 'discount_rate_false_rank'] = l_feat.groupby(keys)['discount_rate'].rank(ascending=False)

    # 商家-满减最低消费正反排序
    l_feat[prefixs + 'min_cost_of_manjian_true_rank'] = l_feat.groupby(keys)['min_cost_of_manjian'].rank(ascending=True)
    l_feat[prefixs + 'min_cost_of_manjian_false_rank'] = l_feat.groupby(keys)['min_cost_of_manjian'].rank(
        ascending=False)

    # 优惠券
    keys = ['Coupon_id']
    prefixs + 'label_field_rank_' + '_'.join(keys) + '_'

    # 优惠券-距离正反排序
    l_feat[prefixs + 'distance_true_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=True)
    l_feat[prefixs + 'distance_false_rank'] = l_feat.groupby(keys)['Distance'].rank(ascending=False)

    # 优惠券-领券日期正反排序
    l_feat[prefixs + 'date_received_true_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=True)
    l_feat[prefixs + 'date_received_false_rank'] = l_feat.groupby(keys)['Date_received'].rank(ascending=False)

    # 填充空值
    l_feat.fillna(0, downcast='infer', inplace=True)

    # 删去'cnt'列
    l_feat.drop(['cnt'], axis=1, inplace=True)

    return l_feat


def get_history_User_feature(history_field, label_field):
    data = history_field.copy()
    data['Date_received'] = data['Date_received'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['cnt'] = 1

    # 用户特征
    keys = ['User_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    u_feat = label_field[keys].drop_duplicates(keep='first')

    # 用户领券数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    # 用户核销数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'nan')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'received_and_cost_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)
    return u_feat


def get_history_Coupon_feature(history_field, label_field):
    data = history_field.copy()
    data['Date_received'] = data['Date_received'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['cnt'] = 1

    # 优惠券特征
    keys = ['Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    c_feat = label_field[keys].drop_duplicates(keep='first')
    return c_feat


def get_all_history_feature(history, label):  # 获得历史区间的所有特征
    u_feature = get_history_User_feature(history, label)
    c_feature = get_history_Coupon_feature(history, label)
    # 将特征链接起来
    h_feat = label.copy()  # 这里已经是算获得标签区间中的所有原来的特征
    h_feat = pd.merge(h_feat, u_feature, on=['User_id'], how='left')
    # h_feat = pd.merge(h_feat, c_feature, on=['Coupon_id'], how='left')

    return h_feat


def get_week_feature(label_field):
    """根据Date_received得到的一些日期特征

    根据date_received列得到领券日是周几,新增一列week存储,并将其one-hot离散为week_0,week_1,week_2,week_3,week_4,week_5,week_6;
    根据week列得到领券日是否为休息日,新增一列is_weekend存储;

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    # 返回的特征数据集
    w_feat = data.copy()
    w_feat['week'] = w_feat['date_received'].map(lambda x: x.weekday())  # 星期几
    w_feat['is_weekend'] = w_feat['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    w_feat = pd.concat([w_feat, pd.get_dummies(w_feat['week'], prefix='week')], axis=1)  # one-hot离散星期几
    w_feat.index = range(len(w_feat))  # 重置index
    # 返回
    return w_feat


def get_dataset(history_field, middle_field, label_field):
    # 特征工程
    label_feat = get_label_feature(label_field)
    history_feat = get_all_history_feature(history_field, label_field)
    week_feat = get_week_feature(label_field)

    # 构造数据集
    share_characters = list(set(label_feat.columns.tolist()) & set(history_feat.columns.tolist()) & set(
        week_feat.columns.tolist()))  # 共有属性,包括id和一些基础特征,为每个特征块的交集
    dataset = pd.concat([week_feat, label_feat.drop(share_characters, axis=1)], axis=1)  # 将两个特征结合起来，删除共同特征
    dataset = pd.concat([dataset, history_feat.drop(share_characters, axis=1)], axis=1)
    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  # 表示训练集和验证集
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset


def model_xgb(train, test):
    """xgb模型

    Args:

    Returns:

    """
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,  # 原5
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,  # 原0.7,用来控制每棵树的随机采样的 列数的占比
              'subsample': 0.9,  # 原0.9,用来控制对于每棵树随机采样比例
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)  # 原5167
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


if __name__ == '__main__':
    # 源数据
    off_train = pd.read_csv('ccf_offline_stage1_train.csv')
    off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)
    # 离散特征
    pd.get_dummies(off_train['Distance'])
    pd.pivot_table(off_train, index='User_id', columns='Discount_rate', values='Distance', aggfunc='count')

    # 划分区间
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    # 保存训练集、验证集、测试集
    train.to_csv('train.csv', index=False, header=None)
    validate.to_csv('validate.csv', index=False, header=None)
    test.to_csv('test.csv', index=False, header=None)

    # 线上训练
    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance = model_xgb(big_train, test)
    # 保存
    result.to_csv('submission.csv', index=False, header=None)



