import pandas as pd
import numpy as np

# 加载数据集
df = pd.read_csv('accepted_filtered_clean.csv')
# 验证关键字段
required_cols = ['label', 'UnRate', 'issue_d']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"数据集缺失关键字段：{missing_cols}，请补充或调整字段名！")
print("✅ label/UnRate/issue_d）均存在")

# 1. 转换时间字段（假设格式为"2018-01"，若不同需调整format参数）
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%Y-%m')
df['issue_year'] = df['issue_d'].dt.year

# 2. 定义场景（二选一，推荐方式1，更贴合宏观事件）
# 方式1：按时间划分（例：2020-2021年疫情为冲击期）
shock_mask = df['issue_year'].between(2008, 2009)
# 方式2：按失业率划分（UnRate>历史均值1.2倍为冲击期，适配无明确事件场景）
# unrate_mean = df['UnRate'].mean()
# shock_mask = df['UnRate'] > unrate_mean * 1.2

# 拆分样本
shock_df = df[shock_mask].copy()  # 冲击期样本
normal_df = df[~shock_mask].copy()  # 正常期样本

# 验证标签格式（确保为0=正常，1=违约）
df['label'] = df['label'].astype(int)
shock_df['label'] = shock_df['label'].astype(int)
normal_df['label'] = normal_df['label'].astype(int)

# 输出样本概况
print(f"📊 冲击期样本：{shock_df.shape[0]}条，违约率：{shock_df['label'].mean():.3f}")
print(f"📊 正常期样本：{normal_df.shape[0]}条，违约率：{normal_df['label'].mean():.3f}")

def clean_dataset(data):
    # 1. 删除标签缺失的样本
    data = data.dropna(subset=['label'])
    # 2. 数值型特征（含UnRate）用中位数填充（抗极端值）
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.drop('label')
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    # 3. 分类型特征用众数填充（若数据含分类型字段，如loan_status）
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
        # 编码分类型特征（转为数值）
        data[col] = data[col].astype('category').cat.codes
    return data

# 清洗两组样本
shock_df_clean = clean_dataset(shock_df)
normal_df_clean = clean_dataset(normal_df)
print("✅ 数据清洗完成（缺失值处理+分类型编码）")

def split_features(data):
    # 1. 基础特征：常规风控特征（排除标签、时间、宏观变量）
    basic_features = [
        'annual_inc',  # 借款人年收入
        'loan_amnt',   # 贷款金额
        'dti',         # 债务收入比
        'fico_range_low',  # 信用评分下限
        'term',        # 贷款期限（已编码）
        'emp_length'   # 就业时长（已编码）
    ]
    # 过滤数据中不存在的基础特征（避免报错）
    basic_features = [col for col in basic_features if col in data.columns]

    # 2. 宏观变量：以UnRate为核心（可补充其他宏观指标，如CPI，若数据有）
    macro_features = ['UnRate']  # 核心宏观变量：失业率
    macro_features = [col for col in macro_features if col in data.columns]

    # 3. 两种特征组合（关键对比组）
    features_no_macro = basic_features  # 未融合宏观变量（仅基础特征）
    features_with_macro = basic_features + macro_features  # 融合宏观变量（基础+UnRate）

    # 输出特征概况
    print(f"\n🔍 基础特征（共{len(basic_features)}个）：{basic_features}")
    print(f"🔍 宏观变量（共{len(macro_features)}个）：{macro_features}")
    print(f"🔍 融合后特征（共{len(features_with_macro)}个）：{features_with_macro}")
    return features_no_macro, features_with_macro

# 划分特征（用正常期数据确认字段，两组样本特征一致）
features_no_macro, features_with_macro = split_features(normal_df_clean)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
# 训练前准备（样本拆分 + 平衡）
def prepare_train_data(data, features):
    # 提取特征和标签
    X = data[features]
    y = data['label']
    # 7:3拆分训练集/测试集（分层抽样，保证违约率一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # SMOTE过采样（仅训练集，避免数据泄露）
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    # 标准化（仅对逻辑回归，XGBoost无需标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_scaled, X_test_scaled, y_train_smote, y_test), scaler

# 模型训练与超参数调优
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_optimized_model(model_type, X_train, y_train):
    # 定义模型与超参数网格
    if model_type == 'lr':  # 逻辑回归
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {'C': [0.01, 0.1, 1, 10]}  # 正则化系数（控制过拟合）
    elif model_type == 'xgb':  # XGBoost
        model = XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='auc')
        param_grid = {
            'learning_rate': [0.01, 0.1],  # 步长
            'max_depth': [3, 5],  # 树深度（控制过拟合）
            'n_estimators': [100, 200]  # 树数量
        }
    else:
        raise ValueError("模型类型仅支持'lr'（逻辑回归）和'xgb'（XGBoost）")

    # 网格搜索调优（5折交叉验证，以AUC为目标）
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_  # 返回最优模型

# 存储所有模型结果（key：场景_模型_特征组合，value：AUC、违约准确率等）
model_results = {}

# 遍历所有场景、模型类型、特征组合（共2场景×2模型×2特征=8组结果）
scenes = [('正常期', normal_df_clean), ('冲击期', shock_df_clean)]
model_types = ['lr', 'xgb']
feature_groups = [
    ('未融合宏观', features_no_macro),
    ('融合宏观(含UnRate)', features_with_macro)
]

#
for scene_name, scene_data in scenes:
    for model_type in model_types:
        for feat_name, features in feature_groups:
            # 1. 准备训练数据
            (X_train, X_test, y_train, y_test), _ = prepare_train_data(scene_data, features)
            # 2. 训练优化模型
            best_model = train_optimized_model(model_type, X_train, y_train)
            # 3. 预测与评估
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 违约概率
            y_pred = best_model.predict(X_test)  # 类别预测
            # 计算核心指标
            auc = roc_auc_score(y_test, y_pred_proba)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            default_acc = tp / (tp + fn) if (tp + fn) > 0 else 0  # 违约预测准确率
            # 存储结果
            result_key = f"{scene_name}_{model_type.upper()}_{feat_name}"
            model_results[result_key] = {
                'AUC': round(auc, 4),
                '违约预测准确率': round(default_acc, 4),
                '模型': best_model
            }
            # 输出训练结果
            print(f"✅ {result_key}：AUC={auc:.4f}，违约准确率={default_acc:.4f}")
            
import pandas as pd

# 1. 整理模型结果为DataFrame
result_list = []
for key, metrics in model_results.items():
    scene, model, feat_type = key.split('_', 2)  # 拆分场景、模型、特征类型
    result_list.append({
        '场景': scene,
        '模型类型': model,
        '特征组合': feat_type,
        'AUC': metrics['AUC'],
        '违约预测准确率': metrics['违约预测准确率']
    })
result_df = pd.DataFrame(result_list)
print("\n📋 所有模型核心指标汇总：")
print(result_df.round(4))

# 2. 计算冲击期AUC下降幅度（核心对比）
drop_analysis = []
for model in ['LR', 'XGB']:
    for feat in ['未融合宏观', '融合宏观(含UnRate)']:
        # 提取正常期和冲击期的AUC
        normal_auc = result_df[
            (result_df['场景'] == '正常期') & 
            (result_df['模型类型'] == model) & 
            (result_df['特征组合'] == feat)
        ]['AUC'].values[0]
        shock_auc = result_df[
            (result_df['场景'] == '冲击期') & 
            (result_df['模型类型'] == model) & 
            (result_df['特征组合'] == feat)
        ]['AUC'].values[0]
        # 计算下降幅度（%）
        auc_drop = round((normal_auc - shock_auc) / normal_auc * 100, 2)
        drop_analysis.append({
            '模型类型': model,
            '特征组合': feat,
            '正常期AUC': normal_auc,
            '冲击期AUC': shock_auc,
            'AUC下降幅度(%)': auc_drop
        })
drop_df = pd.DataFrame(drop_analysis)
print("\n📊 冲击期AUC下降幅度分析（核心结论依据）：")
print(drop_df.round(4))
