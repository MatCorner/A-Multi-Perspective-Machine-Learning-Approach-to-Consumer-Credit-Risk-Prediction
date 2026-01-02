import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Step 1: Data Loading
# ============================================================================

def load_data(file_path):
    """
    Load data file
    Supports CSV and Excel formats
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format, please use CSV file")

    print(f"Data loaded successfully!")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"\nColumn list:\n{df.columns.tolist()}")

    return df


# ============================================================================
# Step 2: Define Default Label
# ============================================================================

def create_default_label(df):
    """
    Create default label
    Charged Off is considered default
    Fully Paid is considered normal
    Current status loans will be excluded (outcome unknown)
    """
    # View all values of loan_status
    print("\nLoan status distribution:")
    print(df['loan_status'].value_counts())

    # Define default status
    default_status = ['Charged Off']
    normal_status = ['Fully Paid']

    # Keep only completed loans (exclude Current and other ongoing statuses)
    df_filtered = df[df['loan_status'].isin(default_status + normal_status)].copy()

    # Create default label
    df_filtered['is_default'] = df_filtered['loan_status'].isin(default_status).astype(int)

    print(f"\nFiltered data rows: {len(df_filtered)}")
    print(f"Default samples: {df_filtered['is_default'].sum()}")
    print(f"Normal samples: {(df_filtered['is_default'] == 0).sum()}")
    print(f"Overall default rate: {df_filtered['is_default'].mean():.2%}")

    return df_filtered


# ============================================================================
# Step 3: Data Cleaning and Preprocessing
# ============================================================================

def clean_and_preprocess(df):
    """
    Data cleaning and preprocessing
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print("Starting data cleaning and preprocessing...")
    print("=" * 60)

    # 1. Process geographic variables
    print("\n1. Processing geographic variables...")
    df['addr_state'] = df['addr_state'].str.strip().str.upper()
    print(f"   Number of states: {df['addr_state'].nunique()}")
    print(f"   Missing state codes: {df['addr_state'].isnull().sum()}")

    # 2. Process annual income
    print("\n2. Processing annual income...")
    print(f"   Original missing values: {df['annual_inc'].isnull().sum()}")
    # Handle outliers (negative values and extreme values)
    df['annual_inc'] = df['annual_inc'].clip(lower=0, upper=df['annual_inc'].quantile(0.99))
    # Fill missing values
    median_income = df['annual_inc'].median()
    df['annual_inc'].fillna(median_income, inplace=True)
    print(f"   Missing values after processing: {df['annual_inc'].isnull().sum()}")
    print(f"   Annual income range: ${df['annual_inc'].min():.0f} - ${df['annual_inc'].max():.0f}")

    # 3. Process debt-to-income ratio (DTI)
    print("\n3. Processing debt-to-income ratio...")
    print(f"   Original missing values: {df['dti'].isnull().sum()}")
    df['dti'] = df['dti'].clip(lower=0, upper=50)
    median_dti = df['dti'].median()
    df['dti'].fillna(median_dti, inplace=True)
    print(f"   Missing values after processing: {df['dti'].isnull().sum()}")
    print(f"   DTI range: {df['dti'].min():.2f}% - {df['dti'].max():.2f}%")

    # 4. Process FICO score
    print("\n4. Processing FICO score...")
    df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    median_fico = df['fico_score'].median()
    df['fico_score'].fillna(median_fico, inplace=True)
    print(f"   FICO score range: {df['fico_score'].min():.0f} - {df['fico_score'].max():.0f}")

    # 5. Process revolving credit utilization
    print("\n5. Processing revolving credit utilization...")
    if df['revol_util'].dtype == 'object':
        df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
    df['revol_util'] = df['revol_util'].clip(0, 100).fillna(df['revol_util'].median())
    print(f"   Revolving credit utilization range: {df['revol_util'].min():.1f}% - {df['revol_util'].max():.1f}%")

    # 6. Process interest rate
    print("\n6. Processing interest rate...")
    if df['int_rate'].dtype == 'object':
        df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
    median_rate = df['int_rate'].median()
    df['int_rate'].fillna(median_rate, inplace=True)
    print(f"   Interest rate range: {df['int_rate'].min():.2f}% - {df['int_rate'].max():.2f}%")

    # 7. Process loan amount
    print("\n7. Processing loan amount...")
    df['loan_amnt'].fillna(df['loan_amnt'].median(), inplace=True)
    print(f"   Loan amount range: ${df['loan_amnt'].min():.0f} - ${df['loan_amnt'].max():.0f}")

    # 8. Process credit inquiry count
    print("\n8. Processing credit inquiry count...")
    df['inq_last_6mths'].fillna(0, inplace=True)
    print(f"   Inquiry count range: {df['inq_last_6mths'].min():.0f} - {df['inq_last_6mths'].max():.0f}")

    # 9. Process employment length
    print("\n9. Processing employment length...")

    def parse_emp_length(x):
        if pd.isna(x):
            return x
        x = str(x).strip()
        if '< 1' in x:
            return 0
        elif '10+' in x:
            return 10
        else:
            # Extract digits
            import re
            match = re.search(r'\d+', x)
            if match:
                return int(match.group())
            return np.nan

    df['emp_length_num'] = df['emp_length'].apply(parse_emp_length)
    median_emp = df['emp_length_num'].median()
    df['emp_length_num'].fillna(median_emp, inplace=True)
    print(f"   Employment length range: {df['emp_length_num'].min():.0f} - {df['emp_length_num'].max():.0f} years")

    # 10. Process loan term
    print("\n10. Processing loan term...")
    df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
    print(f"    Loan term: {df['term_months'].unique()}")

    # 11. Process categorical variables
    print("\n11. Processing categorical variables...")

    # home_ownership
    df['home_ownership'] = df['home_ownership'].fillna('UNKNOWN')
    print(f"    Home ownership types: {df['home_ownership'].unique()}")

    # purpose
    df['purpose'] = df['purpose'].fillna('other')
    print(f"    Number of loan purposes: {df['purpose'].nunique()}")

    # sub_grade
    df['sub_grade'] = df['sub_grade'].fillna('UNKNOWN')
    print(f"    Number of loan sub-grades: {df['sub_grade'].nunique()}")

    print("\nData cleaning completed!")

    return df


# ============================================================================
# Step 4: Feature Engineering
# ============================================================================

def feature_engineering(df):
    """
    Feature engineering: Create new features
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print("Starting feature engineering...")
    print("=" * 60)

    # 1. Loan amount to income ratio
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    print(f"\n1. Created loan-to-income ratio")

    # 2. Calculate state-level statistics
    print(f"\n2. Calculating state-level aggregate features...")
    state_stats = df.groupby('addr_state').agg({
        'annual_inc': ['mean', 'median', 'std'],
        'fico_score': ['mean', 'median', 'std'],
        'dti': ['mean', 'median'],
        'int_rate': ['mean', 'median'],
        'loan_amnt': ['mean', 'median'],
        'is_default': ['mean', 'count']  # State-level default rate and sample size
    }).reset_index()

    # Rename columns
    state_stats.columns = ['addr_state',
                           'state_avg_income', 'state_median_income', 'state_std_income',
                           'state_avg_fico', 'state_median_fico', 'state_std_fico',
                           'state_avg_dti', 'state_median_dti',
                           'state_avg_rate', 'state_median_rate',
                           'state_avg_loan', 'state_median_loan',
                           'state_default_rate', 'state_loan_count']

    # Merge back to original data
    df = df.merge(state_stats, on='addr_state', how='left')

    # 3. Create relative indicators
    print(f"\n3. Creating relative indicators...")
    df['income_vs_state'] = df['annual_inc'] / (df['state_avg_income'] + 1)
    df['fico_vs_state'] = df['fico_score'] / (df['state_avg_fico'] + 1)
    df['dti_vs_state'] = df['dti'] / (df['state_avg_dti'] + 1)
    df['rate_vs_state'] = df['int_rate'] / (df['state_avg_rate'] + 1)

    # 4. Create high-risk flags
    print(f"\n4. Creating risk flags...")
    df['high_dti'] = (df['dti'] > 30).astype(int)
    df['low_fico'] = (df['fico_score'] < 660).astype(int)
    df['high_revol_util'] = (df['revol_util'] > 75).astype(int)
    df['high_inq'] = (df['inq_last_6mths'] > 2).astype(int)

    # 5. Encode categorical variables
    print(f"\n5. Encoding categorical variables...")

    # LabelEncoder for categorical variables
    le_home = LabelEncoder()
    df['home_ownership_encoded'] = le_home.fit_transform(df['home_ownership'])

    le_purpose = LabelEncoder()
    df['purpose_encoded'] = le_purpose.fit_transform(df['purpose'])

    le_grade = LabelEncoder()
    df['sub_grade_encoded'] = le_grade.fit_transform(df['sub_grade'])

    print("\nFeature engineering completed!")
    print(f"Final number of features: {len(df.columns)}")

    return df


# ============================================================================
# Step 5: Exploratory Data Analysis
# ============================================================================

def exploratory_analysis(df, output_dir='output'):
    """
    Exploratory data analysis
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 60)
    print("Starting exploratory data analysis...")
    print("=" * 60)

    # 1. State-level default rate analysis
    print("\n1. Analyzing default rates by state...")
    state_analysis = df.groupby('addr_state').agg({
        'is_default': ['mean', 'count'],
        'annual_inc': 'median',
        'fico_score': 'mean',
        'loan_amnt': 'mean'
    }).reset_index()

    state_analysis.columns = ['state', 'default_rate', 'loan_count',
                              'median_income', 'avg_fico', 'avg_loan']

    # Keep only states with sufficient sample size (at least 100 loans)
    state_analysis = state_analysis[state_analysis['loan_count'] >= 100]
    state_analysis = state_analysis.sort_values('default_rate', ascending=False)

    print(f"\nTop 10 states by default rate:")
    print(state_analysis.head(10).to_string(index=False))

    print(f"\nBottom 10 states by default rate:")
    print(state_analysis.tail(10).to_string(index=False))

    # Save to CSV
    state_analysis.to_csv(f'{output_dir}/state_default_analysis.csv', index=False)

    # 2. Visualize state-level default rates
    plt.figure(figsize=(16, 8))
    top_states = state_analysis.head(20)
    sns.barplot(data=top_states, x='state', y='default_rate', palette='Reds_r')
    plt.title('Top 20 States by Default Rate', fontsize=16, fontweight='bold')
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Default Rate', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/state_default_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   Chart saved: {output_dir}/state_default_rates.png")

    # 3. Relationship between default rate and state-level features
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Default rate vs median income
    axes[0, 0].scatter(state_analysis['median_income'],
                       state_analysis['default_rate'], alpha=0.6)
    axes[0, 0].set_xlabel('Median Income ($)', fontsize=11)
    axes[0, 0].set_ylabel('Default Rate', fontsize=11)
    axes[0, 0].set_title('Default Rate vs Median Income', fontsize=12, fontweight='bold')

    # Default rate vs average FICO
    axes[0, 1].scatter(state_analysis['avg_fico'],
                       state_analysis['default_rate'], alpha=0.6, color='orange')
    axes[0, 1].set_xlabel('Average FICO Score', fontsize=11)
    axes[0, 1].set_ylabel('Default Rate', fontsize=11)
    axes[0, 1].set_title('Default Rate vs Average FICO', fontsize=12, fontweight='bold')

    # Default rate vs average loan amount
    axes[1, 0].scatter(state_analysis['avg_loan'],
                       state_analysis['default_rate'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Average Loan Amount ($)', fontsize=11)
    axes[1, 0].set_ylabel('Default Rate', fontsize=11)
    axes[1, 0].set_title('Default Rate vs Average Loan Amount', fontsize=12, fontweight='bold')

    # Default rate distribution
    axes[1, 1].hist(state_analysis['default_rate'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Default Rate', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Distribution of State Default Rates', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/state_features_vs_default.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Chart saved: {output_dir}/state_features_vs_default.png")

    # 4. Feature correlation analysis
    print("\n2. Analyzing feature correlations...")
    correlation_features = ['annual_inc', 'dti', 'fico_score', 'loan_amnt',
                            'revol_util', 'int_rate', 'emp_length_num',
                            'loan_to_income', 'is_default']

    corr_matrix = df[correlation_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Chart saved: {output_dir}/correlation_matrix.png")

    # 5. Feature comparison between default and non-default
    print("\n3. Comparing features between default and non-default samples...")
    comparison_features = ['annual_inc', 'dti', 'fico_score', 'loan_amnt',
                           'revol_util', 'int_rate']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(comparison_features):
        df.boxplot(column=feature, by='is_default', ax=axes[idx])
        axes[idx].set_title(f'{feature}')
        axes[idx].set_xlabel('Is Default (0=No, 1=Yes)')
        plt.sca(axes[idx])
        plt.xticks([1, 2], ['No', 'Yes'])

    plt.suptitle('Feature Comparison: Default vs Non-Default',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/default_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Chart saved: {output_dir}/default_comparison.png")

    print("\nExploratory analysis completed!")

    return state_analysis


# ============================================================================
# Step 6: Prepare Modeling Data
# ============================================================================

def prepare_modeling_data(df):
    """
    Prepare modeling data
    """
    print("\n" + "=" * 60)
    print("Preparing modeling data...")
    print("=" * 60)

    # Select features
    feature_cols = [
        # Personal financial features
        'annual_inc', 'dti', 'emp_length_num',
        # Credit features
        'fico_score', 'revol_util', 'inq_last_6mths',
        # Loan features
        'loan_amnt', 'int_rate', 'term_months',
        # Derived features
        'loan_to_income',
        # State-level features
        'state_default_rate', 'state_avg_income', 'state_avg_fico',
        'income_vs_state', 'fico_vs_state', 'dti_vs_state', 'rate_vs_state',
        # Risk flags
        'high_dti', 'low_fico', 'high_revol_util', 'high_inq',
        # Encoded categorical variables
        'home_ownership_encoded', 'purpose_encoded', 'sub_grade_encoded'
    ]

    # Check if features exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        print(f"\nWarning: The following features do not exist and will be skipped: {missing_features}")

    print(f"\nNumber of features used: {len(available_features)}")
    print(f"Feature list: {available_features}")

    # Prepare features and labels
    X = df[available_features].copy()
    y = df['is_default'].copy()
    state_info = df[['addr_state']].copy()

    # Check for missing values
    print(f"\nChecking for missing values in features:")
    missing_summary = X.isnull().sum()
    if missing_summary.sum() > 0:
        print(missing_summary[missing_summary > 0])
        # Fill remaining missing values
        X = X.fillna(X.median())
    else:
        print("   No missing values")

    # Data standardization
    print(f"\nPerforming data standardization...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)

    # Split training and test sets
    print(f"\nSplitting training and test sets (80-20)...")
    X_train, X_test, y_train, y_test, state_train, state_test = train_test_split(
        X_scaled, y, state_info, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    print(f"Training set default rate: {y_train.mean():.2%}")
    print(f"Test set default rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, state_train, state_test, scaler, available_features


# ============================================================================
# Step 7: Build Models
# ============================================================================

def build_models(X_train, X_test, y_train, y_test, state_test, output_dir='output'):
    """
    Build and evaluate models
    """
    print("\n" + "=" * 60)
    print("Starting modeling...")
    print("=" * 60)

    models = {}
    predictions = {}

    # 1. Logistic Regression model
    print("\n1. Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = {'pred': lr_pred, 'proba': lr_pred_proba}

    print("   Model training completed")
    print(f"   Training set accuracy: {lr_model.score(X_train, y_train):.4f}")
    print(f"   Test set accuracy: {lr_model.score(X_test, y_test):.4f}")
    print(f"   Test set AUC: {roc_auc_score(y_test, lr_pred_proba):.4f}")

    # 2. Random Forest model
    print("\n2. Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    models['Random Forest'] = rf_model
    predictions['Random Forest'] = {'pred': rf_pred, 'proba': rf_pred_proba}

    print("   Model training completed")
    print(f"   Training set accuracy: {rf_model.score(X_train, y_train):.4f}")
    print(f"   Test set accuracy: {rf_model.score(X_test, y_test):.4f}")
    print(f"   Test set AUC: {roc_auc_score(y_test, rf_pred_proba):.4f}")

    # 3. Model evaluation
    print("\n" + "=" * 60)
    print("Model Evaluation Report")
    print("=" * 60)

    for model_name, preds in predictions.items():
        print(f"\n{model_name}:")
        print("\nClassification Report:")
        print(classification_report(y_test, preds['pred'],
                                    target_names=['Non-Default', 'Default']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, preds['pred'])
        print(cm)

        auc_score = roc_auc_score(y_test, preds['proba'])
        print(f"\nAUC Score: {auc_score:.4f}")

    # 4. ROC curve comparison
    plt.figure(figsize=(10, 8))

    for model_name, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds['proba'])
        auc_score = roc_auc_score(y_test, preds['proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC curves saved: {output_dir}/roc_curves.png")

    # 5. Feature importance analysis (Random Forest)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance Top 15:")
    print(feature_importance.head(15).to_string(index=False))

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 20 Feature Importances (Random Forest)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance chart saved: {output_dir}/feature_importance.png")

    # 6. Analyze model performance by state
    print("\nAnalyzing model performance by state...")
    state_performance = pd.DataFrame({
        'addr_state': state_test['addr_state'],
        'actual': y_test,
        'lr_pred_proba': predictions['Logistic Regression']['proba'],
        'rf_pred_proba': predictions['Random Forest']['proba']
    })

    state_perf_summary = state_performance.groupby('addr_state').agg({
        'actual': ['mean', 'count'],
        'lr_pred_proba': 'mean',
        'rf_pred_proba': 'mean'
    }).reset_index()

    state_perf_summary.columns = ['state', 'actual_default_rate', 'count',
                                  'lr_predicted_rate', 'rf_predicted_rate']

    state_perf_summary = state_perf_summary[state_perf_summary['count'] >= 20]
    state_perf_summary['lr_error'] = abs(state_perf_summary['actual_default_rate'] -
                                         state_perf_summary['lr_predicted_rate'])
    state_perf_summary['rf_error'] = abs(state_perf_summary['actual_default_rate'] -
                                         state_perf_summary['rf_predicted_rate'])

    state_perf_summary = state_perf_summary.sort_values('actual_default_rate', ascending=False)

    print("\nActual vs Predicted Default Rates by State:")
    print(state_perf_summary.head(15).to_string(index=False))

    state_perf_summary.to_csv(f'{output_dir}/state_model_performance.csv', index=False)

    return models, predictions, feature_importance, state_perf_summary


# ============================================================================
# Main Function: Complete Workflow
# ============================================================================

def main(file_path, output_dir='output'):
    """
    Complete data analysis workflow
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 80)
    print(" " * 15 + "Analysis of Geographic Factors on Default Rate")
    print("=" * 80)

    # Step 1: Load data
    df = load_data(file_path)

    # Step 2: Create default label
    df = create_default_label(df)

    # Step 3: Data cleaning and preprocessing
    df = clean_and_preprocess(df)

    # Step 4: Feature engineering
    df = feature_engineering(df)

    # Step 5: Exploratory data analysis
    state_analysis = exploratory_analysis(df, output_dir)

    # Step 6: Prepare modeling data
    X_train, X_test, y_train, y_test, state_train, state_test, scaler, features = \
        prepare_modeling_data(df)

    # Step 7: Build and evaluate models
    models, predictions, feature_importance, state_performance = \
        build_models(X_train, X_test, y_train, y_test, state_test, output_dir)

    print("\n" + "=" * 80)
    print(" " * 25 + "Analysis Completed!")
    print("=" * 80)
    print(f"\nAll results saved to directory: {output_dir}/")
    print("\nGenerated files:")
    print("  - state_default_analysis.csv: State default rate statistics")
    print("  - state_model_performance.csv: State model performance")
    print("  - state_default_rates.png: State-level default rate chart")
    print("  - state_features_vs_default.png: State features vs default rate relationship")
    print("  - correlation_matrix.png: Feature correlation matrix")
    print("  - default_comparison.png: Default vs non-default feature comparison")
    print("  - roc_curves.png: ROC curves comparison")
    print("  - feature_importance.png: Feature importance chart")

    return df, models, state_analysis, state_performance


if __name__ == "__main__":
    # Data file path
    file_path = r"C:\Users\18345\Desktop\6013\project\accepted_2007_to_2018Q4.csv"

    # Run complete analysis
    df, models, state_analysis, state_performance = main(file_path, output_dir='output')
