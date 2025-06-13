import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib
from sklearn.impute import SimpleImputer

# Load the dataset
url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/employee_data-00NHa6uF33gD5YcrVLrRq7o08XPJVa.csv"
df = pd.read_csv(url)

# Display basic information
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Convert data types if needed
# Convert Attrition to numeric (0 or 1)
if df['Attrition'].dtype == 'object':
    df['Attrition'] = df['Attrition'].astype(float)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Attrition rate calculation
attrition_rate = df['Attrition'].mean() * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

# Exploratory Data Analysis

# 1. Attrition by Department
dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False) * 100
print("\nAttrition Rate by Department:")
print(dept_attrition)

# 2. Attrition by Job Role
role_attrition = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False) * 100
print("\nAttrition Rate by Job Role:")
print(role_attrition)

# 3. Attrition by Age Group
df['AgeGroup'] = pd.cut(df['Age'].astype(float), bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
age_attrition = df.groupby('AgeGroup')['Attrition'].mean().sort_values(ascending=False) * 100
print("\nAttrition Rate by Age Group:")
print(age_attrition)

# 4. Attrition by Overtime
ot_attrition = df.groupby('OverTime')['Attrition'].mean().sort_values(ascending=False) * 100
print("\nAttrition Rate by Overtime:")
print(ot_attrition)

# 5. Attrition by Job Satisfaction
satisfaction_attrition = df.groupby('JobSatisfaction')['Attrition'].mean().sort_values(ascending=False) * 100
print("\nAttrition Rate by Job Satisfaction:")
print(satisfaction_attrition)

# 6. Correlation with Monthly Income
income_corr = df.groupby('MonthlyIncome')['Attrition'].mean()
print("\nCorrelation between Monthly Income and Attrition:")
print(f"Correlation: {df['MonthlyIncome'].astype(float).corr(df['Attrition']):.4f}")

# Prepare data for modeling
# Separate features and target
X = df.drop(['Attrition', 'EmployeeId', 'EmployeeCount', 'StandardHours', 'Over18', 'AgeGroup'], axis=1, errors='ignore')
y = df['Attrition']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print("\nCategorical Features:", categorical_cols)
print("Numerical Features:", numerical_cols)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create and evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_names = numerical_cols + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols))
feature_importance = model.named_steps['classifier'].feature_importances_

# Sort feature importances
sorted_idx = np.argsort(feature_importance)[::-1]
top_10_idx = sorted_idx[:10]

print("\nTop 10 Important Features:")
for i in top_10_idx:
    if i < len(feature_names):
        print(f"{feature_names[i]}: {feature_importance[i]:.4f}")

# Save the model
joblib.dump(model, 'hr_attrition_model.pkl')
print("\nModel saved as 'hr_attrition_model.pkl'")

# Generate visualizations for the dashboard
plt.figure(figsize=(12, 6))

# 1. Attrition by Department
plt.subplot(2, 3, 1)
sns.barplot(x=dept_attrition.index, y=dept_attrition.values)
plt.title('Attrition by Department')
plt.xticks(rotation=45)
plt.ylabel('Attrition Rate (%)')

# 2. Attrition by Job Role
plt.subplot(2, 3, 2)
top_roles = role_attrition.head(5)
sns.barplot(x=top_roles.index, y=top_roles.values)
plt.title('Top 5 Job Roles by Attrition')
plt.xticks(rotation=45)
plt.ylabel('Attrition Rate (%)')

# 3. Attrition by Age Group
plt.subplot(2, 3, 3)
sns.barplot(x=age_attrition.index, y=age_attrition.values)
plt.title('Attrition by Age Group')
plt.ylabel('Attrition Rate (%)')

# 4. Attrition by Overtime
plt.subplot(2, 3, 4)
sns.barplot(x=ot_attrition.index, y=ot_attrition.values)
plt.title('Attrition by Overtime')
plt.ylabel('Attrition Rate (%)')

# 5. Attrition by Job Satisfaction
plt.subplot(2, 3, 5)
sns.barplot(x=satisfaction_attrition.index.astype(str), y=satisfaction_attrition.values)
plt.title('Attrition by Job Satisfaction')
plt.xlabel('Satisfaction Level')
plt.ylabel('Attrition Rate (%)')

# 6. Feature Importance
plt.subplot(2, 3, 6)
top_5_idx = sorted_idx[:5]
top_features = [feature_names[i] if i < len(feature_names) else "Unknown" for i in top_5_idx]
top_importance = [feature_importance[i] for i in top_5_idx]
sns.barplot(x=top_importance, y=top_features)
plt.title('Top 5 Features for Attrition')
plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('attrition_dashboard.png')
print("\nDashboard visualization saved as 'attrition_dashboard.png'")
