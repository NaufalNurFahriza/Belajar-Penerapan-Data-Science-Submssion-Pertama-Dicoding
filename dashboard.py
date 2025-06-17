import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard - Jaya Jaya Maju",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize matplotlib to use Agg backend for better compatibility
plt.switch_backend('Agg')

# Load data and model with better error handling
@st.cache_data
def load_data():
    """Load the dataset with proper error handling"""
    file_path = "dataset/attrition_dashboard_data.csv"
    if not os.path.exists(file_path):
        # Try alternative paths
        alternative_paths = [
            "attrition_dashboard_data.csv",
            "./dataset/attrition_dashboard_data.csv",
            "../dataset/attrition_dashboard_data.csv"
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                file_path = path
                break
        else:
            raise FileNotFoundError(f"Dataset file not found. Looked in: {[file_path] + alternative_paths}")
    
    return pd.read_csv(file_path)

@st.cache_resource
def load_model():
    """Load the trained model with proper error handling"""
    model_path = "model/attrition_model.pkl"
    if not os.path.exists(model_path):
        # Try alternative paths
        alternative_paths = [
            "attrition_model.pkl",
            "./model/attrition_model.pkl",
            "../model/attrition_model.pkl"
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            st.warning("Model file not found. Prediction functionality will be limited.")
            return None
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}. Prediction functionality will be limited.")
        return None

# Load data with better error handling
try:
    df = load_data()
    model = load_model()
    
    # Ensure required columns exist
    required_columns = ['Age', 'MonthlyIncome', 'Attrition', 'Department']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns in dataset: {missing_columns}")
        st.stop()
        
except FileNotFoundError as e:
    st.error(f"Error: {str(e)}")
    st.info("Please ensure your dataset file is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading data: {str(e)}")
    st.stop()

# Feature lists - make them dynamic based on available columns
available_columns = df.columns.tolist()

numeric_features = [col for col in [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
] if col in available_columns]

ordinal_cols = [col for col in [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'WorkLifeBalance'
] if col in available_columns]

nominal_cols = [col for col in [
    'BusinessTravel', 'Department', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'OverTime'
] if col in available_columns]

# Custom labels for ordinal features
ordinal_label_map = {
    "WorkLifeBalance": {
        1: "Poor", 2: "Fair", 3: "Good", 4: "Excellent"
    },
    "PerformanceRating": {
        1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"
    },
    "JobSatisfaction": {
        1: "Very Dissatisfied", 2: "Dissatisfied", 3: "Satisfied", 4: "Very Satisfied"
    },
    "Education": {
        1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"
    }
}

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #1f77b4;">🏢 Jaya Jaya Maju</h2>
        <p style="color: #666;">HR Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["📊 Dashboard Overview", "🔍 Feature Analysis", "🤖 Attrition Predictor", "💡 Insights"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Data Filters")
    
    # Safe filter creation based on available columns
    if 'Department' in df.columns:
        department_filter = st.multiselect(
            "Filter by Department",
            options=df['Department'].unique(),
            default=df['Department'].unique()
        )
    else:
        department_filter = []
    
    if 'Attrition' in df.columns:
        attrition_filter = st.selectbox(
            "Filter by Attrition Status",
            options=["All", "Attrited", "Not Attrited"],
            index=0
        )
    else:
        attrition_filter = "All"
    
    if 'Age' in df.columns:
        age_range = st.slider(
            "Filter by Age Range",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max()))
        )
    else:
        age_range = (18, 65)

# Apply filters safely
filtered_df = df.copy()

if department_filter and 'Department' in df.columns:
    filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]

if 'Attrition' in df.columns:
    if attrition_filter == "Attrited":
        filtered_df = filtered_df[filtered_df['Attrition'] == 1]
    elif attrition_filter == "Not Attrited":
        filtered_df = filtered_df[filtered_df['Attrition'] == 0]

if 'Age' in df.columns:
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

# Dashboard Overview
if menu == "📊 Dashboard Overview":
    st.title("HR Attrition Analytics Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(filtered_df))
    
    with col2:
        if 'Attrition' in df.columns:
            attrition_rate = filtered_df['Attrition'].mean() * 100
            overall_rate = df['Attrition'].mean() * 100
            delta = attrition_rate - overall_rate
            st.metric(
                "Attrition Rate", 
                f"{attrition_rate:.1f}%",
                delta=f"{delta:+.1f}% vs overall",
                help="Percentage of employees who left the company"
            )
        else:
            st.metric("Attrition Rate", "N/A")
    
    with col3:
        if 'YearsAtCompany' in df.columns:
            avg_tenure = filtered_df['YearsAtCompany'].mean()
            st.metric("Avg Company Tenure", f"{avg_tenure:.1f} years")
        else:
            st.metric("Avg Company Tenure", "N/A")
    
    with col4:
        if 'MonthlyIncome' in df.columns:
            avg_income = filtered_df['MonthlyIncome'].mean()
            st.metric("Avg Monthly Income", f"${avg_income:,.0f}")
        else:
            st.metric("Avg Monthly Income", "N/A")
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Department' in df.columns and 'Attrition' in df.columns:
            st.subheader("Attrition by Department")
            try:
                attrition_by_dept = filtered_df.groupby(['Department', 'Attrition']).size().unstack().fillna(0)
                if 1 in attrition_by_dept.columns and 0 in attrition_by_dept.columns:
                    attrition_by_dept['Attrition Rate'] = attrition_by_dept[1] / (attrition_by_dept[0] + attrition_by_dept[1]) * 100
                    
                    fig = px.bar(
                        attrition_by_dept.reset_index(),
                        x='Department',
                        y='Attrition Rate',
                        color='Department',
                        text='Attrition Rate',
                        labels={'Attrition Rate': 'Attrition Rate (%)'},
                        height=400
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for attrition analysis by department")
            except Exception as e:
                st.error(f"Error creating department chart: {str(e)}")
        else:
            st.info("Department or Attrition data not available")
    
    with col2:
        if 'MonthlyIncome' in df.columns and 'Attrition' in df.columns:
            st.subheader("Monthly Income Distribution")
            try:
                fig = px.box(
                    filtered_df,
                    x='Attrition',
                    y='MonthlyIncome',
                    color='Attrition',
                    points="all",
                    labels={'MonthlyIncome': 'Monthly Income ($)', 'Attrition': 'Attrition Status'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating income chart: {str(e)}")
        else:
            st.info("Income or Attrition data not available")
    
    st.markdown("---")
    
    # Additional charts
    st.subheader("Key Factors Analysis")
    
    # Create tabs dynamically based on available data
    available_tabs = []
    if 'WorkLifeBalance' in df.columns:
        available_tabs.append("Work-Life Balance")
    if 'JobSatisfaction' in df.columns:
        available_tabs.append("Job Satisfaction")
    if 'OverTime' in df.columns:
        available_tabs.append("Overtime")
    if 'Age' in df.columns:
        available_tabs.append("Age Distribution")
    
    if available_tabs:
        tabs = st.tabs(available_tabs)
        
        tab_idx = 0
        if 'WorkLifeBalance' in df.columns and 'Attrition' in df.columns:
            with tabs[tab_idx]:
                try:
                    wlb_counts = filtered_df.groupby(['WorkLifeBalance', 'Attrition']).size().unstack().fillna(0)
                    wlb_counts = wlb_counts.div(wlb_counts.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(
                        wlb_counts.reset_index(),
                        x='WorkLifeBalance',
                        y=[0, 1] if 0 in wlb_counts.columns and 1 in wlb_counts.columns else wlb_counts.columns.tolist(),
                        barmode='group',
                        labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
                        title='Attrition by Work-Life Balance Rating',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating work-life balance chart: {str(e)}")
            tab_idx += 1
        
        if 'JobSatisfaction' in df.columns and 'Attrition' in df.columns and tab_idx < len(tabs):
            with tabs[tab_idx]:
                try:
                    js_counts = filtered_df.groupby(['JobSatisfaction', 'Attrition']).size().unstack().fillna(0)
                    js_counts = js_counts.div(js_counts.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(
                        js_counts.reset_index(),
                        x='JobSatisfaction',
                        y=[0, 1] if 0 in js_counts.columns and 1 in js_counts.columns else js_counts.columns.tolist(),
                        barmode='group',
                        labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
                        title='Attrition by Job Satisfaction Rating',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating job satisfaction chart: {str(e)}")
            tab_idx += 1
        
        if 'OverTime' in df.columns and 'Attrition' in df.columns and tab_idx < len(tabs):
            with tabs[tab_idx]:
                try:
                    ot_counts = filtered_df.groupby(['OverTime', 'Attrition']).size().unstack().fillna(0)
                    ot_counts = ot_counts.div(ot_counts.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(
                        ot_counts.reset_index(),
                        x='OverTime',
                        y=[0, 1] if 0 in ot_counts.columns and 1 in ot_counts.columns else ot_counts.columns.tolist(),
                        barmode='group',
                        labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
                        title='Attrition by Overtime Status',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating overtime chart: {str(e)}")
            tab_idx += 1
        
        if 'Age' in df.columns and 'Attrition' in df.columns and tab_idx < len(tabs):
            with tabs[tab_idx]:
                try:
                    fig = px.histogram(
                        filtered_df,
                        x='Age',
                        color='Attrition',
                        nbins=20,
                        barmode='overlay',
                        opacity=0.7,
                        labels={'Age': 'Age (years)'},
                        title='Age Distribution by Attrition Status',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating age distribution chart: {str(e)}")
    else:
        st.info("No additional analysis data available")

# Feature Analysis
elif menu == "🔍 Feature Analysis":
    st.title("Feature Analysis")
    
    st.subheader("Feature Importance")
    if model is not None:
        try:
            if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
                feat_names = model.named_steps['preprocessor'].get_feature_names_out()
                importances = model.named_steps['classifier'].feature_importances_
                imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
                imp_df = imp_df.sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(
                    imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Important Features for Attrition Prediction',
                    labels={'Importance': 'Feature Importance', 'Feature': ''},
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance not available for this model type.")
        except Exception as e:
            st.error(f"Error displaying feature importance: {str(e)}")
    else:
        st.warning("Model not loaded. Feature importance analysis unavailable.")
    
    st.markdown("---")
    
    st.subheader("Feature Comparison Tool")
    
    # Get available features excluding target
    available_features = [col for col in df.columns if col != 'Attrition']
    
    if available_features:
        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox(
                "Select feature to analyze",
                options=sorted(available_features)
            )
        
        with col2:
            if feature in numeric_features:
                agg_func = st.selectbox(
                    "Aggregation",
                    options=["Mean", "Median", "Sum", "Count"],
                    index=0
                )
        
        if feature in numeric_features and 'Attrition' in df.columns:
            try:
                agg_df = filtered_df.groupby('Attrition')[feature].agg(
                    Mean='mean',
                    Median='median',
                    Sum='sum',
                    Count='count'
                ).reset_index()
                
                fig = px.bar(
                    agg_df,
                    x='Attrition',
                    y=agg_func,
                    color='Attrition',
                    text=agg_func,
                    title=f'{agg_func} {feature} by Attrition Status',
                    labels={agg_func: agg_func, 'Attrition': 'Attrition Status'},
                    height=400
                )
                if agg_func in ['Mean', 'Median']:
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                else:
                    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = px.box(
                    filtered_df,
                    x='Attrition',
                    y=feature,
                    color='Attrition',
                    points="all",
                    title=f'Distribution of {feature} by Attrition Status',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating feature analysis: {str(e)}")
        
        elif 'Attrition' in df.columns:
            try:
                count_df = filtered_df.groupby([feature, 'Attrition']).size().unstack().fillna(0)
                count_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    count_df.reset_index(),
                    x=feature,
                    y=[0, 1] if 0 in count_df.columns and 1 in count_df.columns else count_df.columns.tolist(),
                    barmode='group',
                    labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
                    title=f'Attrition by {feature}',
                    height=500
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating categorical analysis: {str(e)}")
    else:
        st.info("No features available for analysis")

# Attrition Predictor
elif menu == "🤖 Attrition Predictor":
    st.title("Employee Attrition Predictor")
    st.markdown("Predict whether an employee is at risk of leaving the company")
    
    if model is None:
        st.warning("⚠️ Prediction model is not available. Please ensure the model file exists.")
        st.info("You can still explore the dashboard insights while the model is unavailable.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Details")
                age = st.slider("Age", 18, 60, 30)
                monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
                total_working_years = st.number_input("Total Working Years", 0, 40, 5)
                overtime = st.radio("Works Overtime", ["No", "Yes"], index=0)
            
            with col2:
                st.subheader("Job Details")
                job_level = st.select_slider("Job Level", options=[1, 2, 3, 4, 5], value=2)
                
                # Job satisfaction with better labels if available
                if 'JobSatisfaction' in ordinal_label_map:
                    job_satisfaction = st.select_slider(
                        "Job Satisfaction", 
                        options=[1, 2, 3, 4],
                        format_func=lambda x: ordinal_label_map['JobSatisfaction'].get(x, str(x))
                    )
                else:
                    job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
                
                years_at_company = st.number_input("Years at Company", 0, 40, 2)
                
                if 'Department' in df.columns:
                    department = st.selectbox("Department", df['Department'].unique())
                else:
                    department = "Unknown"
            
            submitted = st.form_submit_button("Predict Attrition Risk")
            
            if submitted:
                try:
                    # Create input dataframe with safe defaults
                    input_data = pd.DataFrame([{
                        'Age': age,
                        'MonthlyIncome': monthly_income,
                        'TotalWorkingYears': total_working_years,
                        'OverTime': overtime,
                        'JobLevel': job_level,
                        'JobSatisfaction': job_satisfaction,
                        'YearsAtCompany': years_at_company,
                        'Department': department,
                        # Safe defaults for other required features
                        'DailyRate': df['DailyRate'].median() if 'DailyRate' in df.columns else 800,
                        'DistanceFromHome': df['DistanceFromHome'].median() if 'DistanceFromHome' in df.columns else 9,
                        'HourlyRate': df['HourlyRate'].median() if 'HourlyRate' in df.columns else 65,
                        'Education': 3,
                        'EnvironmentSatisfaction': 3,
                        'JobInvolvement': 3,
                        'PerformanceRating': 3,
                        'RelationshipSatisfaction': 3,
                        'StockOptionLevel': 1,
                        'WorkLifeBalance': 3,
                        'BusinessTravel': 'Travel_Rarely',
                        'EducationField': 'Life Sciences',
                        'Gender': 'Male',
                        'JobRole': 'Sales Executive',
                        'MaritalStatus': 'Single',
                        'NumCompaniesWorked': 2,
                        'PercentSalaryHike': 15,
                        'TrainingTimesLastYear': 2,
                        'YearsInCurrentRole': max(0, years_at_company - 1),
                        'YearsSinceLastPromotion': max(0, years_at_company - 1),
                        'YearsWithCurrManager': max(0, years_at_company - 1),
                        'MonthlyRate': monthly_income * 1.2
                    }])
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    prediction_proba = model.predict_proba(input_data)[0]
                    
                    # Display prediction result
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction Result")
                        if prediction == 1:
                            st.error(f"🚨 High Attrition Risk ({prediction_proba[1]*100:.1f}%)")
                        else:
                            st.success(f"✅ Low Attrition Risk ({prediction_proba[0]*100:.1f}%)")
                        
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prediction_proba[1]*100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Attrition Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prediction_proba[1]*100
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Recommended Actions")
                        if prediction == 1:
                            st.markdown(f"""
                            **Immediate Actions:**
                            - Schedule one-on-one retention interview
                            - Review compensation package (current: ${monthly_income:,.0f})
                            - Assess workload and overtime (current: {overtime})
                            
                            **Preventive Measures:**
                            - Offer career development plan
                            - Consider role rotation or promotion
                            - Assign mentor from senior staff
                            """)
                        else:
                            st.markdown(f"""
                            **Maintenance Actions:**
                            - Continue regular check-ins
                            - Monitor career development opportunities
                            - Ensure competitive compensation (current: ${monthly_income:,.0f})
                            
                            **Preventive Measures:**
                            - Annual retention risk assessment
                            - Leadership development program
                            - Flexible work arrangement options
                            """)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Please check that all required fields are filled correctly.")

# Insights
elif menu == "💡 Insights":
    st.title("Key Insights & Recommendations")
    
    if 'Attrition' in df.columns:
        st.subheader("Top Attrition Drivers")
        st.markdown("""
        Our analysis identified these as the strongest predictors of employee attrition:
        """)
        
        # Calculate actual statistics from the data
        drivers = []
        
        if 'MonthlyIncome' in df.columns:
            low_income_attrition = df[df['MonthlyIncome'] < df['MonthlyIncome'].median()]['Attrition'].mean()
            high_income_attrition = df[df['MonthlyIncome'] >= df['MonthlyIncome'].median()]['Attrition'].mean()
            if high_income_attrition > 0:
                income_ratio = low_income_attrition / high_income_attrition
            else:
                income_ratio = "N/A"
            drivers.append({
                "factor": "Monthly Income", 
                "impact": "High", 
                "description": f"Lower income employees have {income_ratio:.1f}x higher attrition rate" if isinstance(income_ratio, float) else "Income analysis not available"
            })
        
        if 'OverTime' in df.columns:
            overtime_attrition = df[df['OverTime'] == 'Yes']['Attrition'].mean() if 'Yes' in df['OverTime'].values else 0
            no_overtime_attrition = df[df['OverTime'] == 'No']['Attrition'].mean() if 'No' in df['OverTime'].values else 0
            if no_overtime_attrition > 0:
                overtime_ratio = overtime_attrition / no_overtime_attrition
            else:
                overtime_ratio = "N/A"
            drivers.append({
                "factor": "Overtime Status", 
                "impact": "High", 
                "description": f"Employees working overtime have {overtime_ratio:.1f}x higher attrition risk" if isinstance(overtime_ratio, float) else "Overtime analysis not available"
            })
        
        if 'Age' in df.columns:
            young_attrition = df[df['Age'] < 30]['Attrition'].mean()
            older_attrition = df[df['Age'] >= 30]['Attrition'].mean()
            if older_attrition > 0:
                age_ratio = young_attrition / older_attrition
            else:
                age_ratio = "N/A"
            drivers.append({
                "factor": "Age", 
                "impact": "Medium", 
                "description": f"Employees under 30 show {age_ratio:.1f}x higher turnover rates" if isinstance(age_ratio, float) else "Age analysis not available"
            })
        
        if 'JobLevel' in df.columns:
            entry_level_attrition = df[df['JobLevel'] <= 2]['Attrition'].mean()
            senior_level_attrition = df[df['JobLevel'] > 2]['Attrition'].mean()
            if senior_level_attrition > 0:
                level_ratio = entry_level_attrition / senior_level_attrition
            else:
                level_ratio = "N/A"
            drivers.append({
                "factor": "Job Level", 
                "impact": "Medium", 
                "description": f"Entry-level positions experience {level_ratio:.1f}x more attrition" if isinstance(level_ratio, float) else "Job level analysis not available"
            })
        
        if 'WorkLifeBalance' in df.columns:
            poor_wlb_attrition = df[df['WorkLifeBalance'] <= 2]['Attrition'].mean()
            good_wlb_attrition = df[df['WorkLifeBalance'] > 2]['Attrition'].mean()
            if good_wlb_attrition > 0:
                wlb_ratio = poor_wlb_attrition / good_wlb_attrition
            else:
                wlb_ratio = "N/A"
            drivers.append({
                "factor": "Work-Life Balance", 
                "impact": "Medium", 
                "description": f"Poor ratings correlate with {wlb_ratio:.1f}x higher attrition" if isinstance(wlb_ratio, float) else "Work-life balance analysis not available"
            })
        
        for driver in drivers:
            with st.expander(f"{driver['factor']} ({driver['impact']} impact)"):
                st.markdown(driver['description'])
                
                try:
                    if driver['factor'] == "Monthly Income" and 'MonthlyIncome' in df.columns:
                        fig = px.box(
                            df,
                            x='Attrition',
                            y='MonthlyIncome',
                            color='Attrition',
                            title='Monthly Income Distribution by Attrition Status'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif driver['factor'] == "Overtime Status" and 'OverTime' in df.columns:
                        ot_data = df.groupby('OverTime')['Attrition'].mean().reset_index()
                        fig = px.bar(
                            ot_data,
                            x='OverTime',
                            y='Attrition',
                            color='OverTime',
                            title='Attrition Rate by Overtime Status',
                            labels={'Attrition': 'Attrition Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif driver['factor'] == "Age" and 'Age' in df.columns:
                        age_bins = pd.cut(df['Age'], bins=[0, 25, 30, 35, 40, 50, 100], labels=['<25', '25-29', '30-34', '35-39', '40-49', '50+'])
                        age_data = df.groupby(age_bins)['Attrition'].mean().reset_index()
                        fig = px.bar(
                            age_data,
                            x='Age',
                            y='Attrition',
                            title='Attrition Rate by Age Group',
                            labels={'Attrition': 'Attrition Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif driver['factor'] == "Job Level" and 'JobLevel' in df.columns:
                        level_data = df.groupby('JobLevel')['Attrition'].mean().reset_index()
                        fig = px.bar(
                            level_data,
                            x='JobLevel',
                            y='Attrition',
                            title='Attrition Rate by Job Level',
                            labels={'Attrition': 'Attrition Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif driver['factor'] == "Work-Life Balance" and 'WorkLifeBalance' in df.columns:
                        wlb_data = df.groupby('WorkLifeBalance')['Attrition'].mean().reset_index()
                        fig = px.bar(
                            wlb_data,
                            x='WorkLifeBalance',
                            y='Attrition',
                            title='Attrition Rate by Work-Life Balance Rating',
                            labels={'Attrition': 'Attrition Rate'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error creating chart for {driver['factor']}: {str(e)}")
        
        st.markdown("---")
        
        st.subheader("Department-Specific Findings")
        if 'Department' in df.columns:
            try:
                dept_data = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False).reset_index()
                
                fig = px.bar(
                    dept_data,
                    x='Department',
                    y='Attrition',
                    color='Department',
                    title='Attrition Rate by Department',
                    labels={'Attrition': 'Attrition Rate'},
                    text='Attrition'
                )
                fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Department analysis
                col1, col2 = st.columns(2)
                
                departments = dept_data['Department'].tolist()
                if len(departments) >= 3:
                    with col1:
                        highest_dept = departments[0]
                        highest_rate = dept_data.iloc[0]['Attrition']
                        
                        st.markdown(f"""
                        **{highest_dept} Department:**
                        - Highest overall attrition rate ({highest_rate:.1%})
                        - Primary drivers need investigation
                        - Recommended: Exit interview analysis
                        - Focus on retention strategies
                        """)
                        
                        if len(departments) >= 2:
                            second_dept = departments[1]
                            second_rate = dept_data.iloc[1]['Attrition']
                            st.markdown(f"""
                            **{second_dept} Department:**
                            - Moderate attrition rate ({second_rate:.1%})
                            - Monitor key risk factors
                            - Preventive measures recommended
                            """)
                    
                    with col2:
                        lowest_dept = departments[-1]
                        lowest_rate = dept_data.iloc[-1]['Attrition']
                        
                        st.markdown(f"""
                        **{lowest_dept} Department:**
                        - Lowest attrition rate ({lowest_rate:.1%})
                        - Best practices to be shared
                        - Strong retention factors present
                        - Model for other departments
                        """)
                
            except Exception as e:
                st.error(f"Error creating department analysis: {str(e)}")
        else:
            st.info("Department data not available for analysis")
    
    else:
        st.info("Attrition data not available for insights generation")
    
    st.markdown("---")
    
    st.subheader("Actionable Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Immediate Priorities", "Medium-Term Initiatives", "Long-Term Strategy"])
    
    with tab1:
        st.markdown("""
        **Within 30 Days:**
        
        - 🔍 Implement overtime monitoring dashboard with automatic alerts when employees exceed 10% overtime monthly
        - 💰 Identify all employees earning below department median and initiate compensation reviews
        - 🗣️ Launch "Stay Interviews" program for high-risk employees (focus on employees with low satisfaction scores)
        - 📊 Set up weekly attrition risk reporting for management team
        
        **Within 90 Days:**
        
        - 🎯 Develop department-specific retention bonus programs based on attrition rates
        - ⚖️ Create task force to address work-life balance concerns, especially for high-overtime departments
        - 📋 Implement comprehensive exit interview analysis process with structured feedback loops
        - 👥 Train managers on retention conversation techniques
        """)
    
    with tab2:
        st.markdown("""
        **6-12 Month Initiatives:**
        
        - **Career Path Development:**
          - Create transparent promotion frameworks for all roles and job levels
          - Implement skills-based progression tracks with clear milestones
          - Launch internal mobility program to reduce external attrition
          - Establish mentorship programs pairing senior and junior employees
        
        - **Manager Training & Development:**
          - Retention leadership certification program for all people managers
          - Coaching workshops for career development conversations
          - Workload management and delegation training
          - Recognition and feedback delivery skills
        
        - **Compensation & Benefits Strategy:**
          - Comprehensive market benchmarking analysis across all roles
          - Performance-based bonus structure implementation
          - Enhanced benefits package review (flexible work, wellness programs)
          - Equity and long-term incentive programs for key talent
        """)
    
    with tab3:
        st.markdown("""
        **Strategic Initiatives (12+ Months):**
        
        - **Advanced Predictive Analytics:**
          - Real-time attrition risk scoring integrated with HRIS systems
          - Early warning system with automated intervention recommendations
          - Machine learning model continuous improvement and recalibration
          - Predictive workforce planning based on attrition patterns
        
        - **Culture & Engagement Transformation:**
          - Quarterly pulse surveys with AI-powered sentiment analysis
          - Employee experience journey mapping and optimization
          - Values alignment programs and culture assessment tools
          - Employee resource groups and inclusion initiatives
        
        - **Strategic Workforce Planning:**
          - Comprehensive skills gap analysis and future workforce modeling
          - Succession planning integration with performance management
          - Talent pipeline development and university partnerships
          - Remote work and hybrid policies optimization
          
        - **Technology & Innovation:**
          - AI-powered exit interview analysis and trend identification
          - Chatbot for employee feedback and career guidance
          - Integration with external job market data for competitive intelligence
          - Blockchain-based credential and achievement tracking
        """)

# Data Summary Section
st.markdown("---")
st.subheader("📈 Data Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Dataset Overview**")
    st.write(f"• Total Records: {len(df):,}")
    st.write(f"• Total Features: {len(df.columns)}")
    if 'Attrition' in df.columns:
        st.write(f"• Overall Attrition Rate: {df['Attrition'].mean():.1%}")

with col2:
    st.markdown("**Available Features**")
    st.write(f"• Numeric Features: {len(numeric_features)}")
    st.write(f"• Categorical Features: {len(nominal_cols)}")
    st.write(f"• Ordinal Features: {len(ordinal_cols)}")

with col3:
    st.markdown("**Model Status**")
    if model is not None:
        st.success("✅ Model Loaded Successfully")
        st.write("• Predictions Available")
        st.write("• Feature Importance Available")
    else:
        st.warning("⚠️ Model Not Available")
        st.write("• Dashboard Analysis Only")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #666;
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
    border-top: 1px solid #eee;
    background-color: #f8f9fa;
    border-radius: 10px;
}
.footer-title {
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
</style>
<div class="footer">
    <div class="footer-title">HR Analytics Dashboard - Jaya Jaya Maju</div>
    Powered by Streamlit • Advanced Analytics for Strategic HR Decisions<br>
    <small>Dashboard Version 2.0 • Enhanced Error Handling & Compatibility</small><br>
    <small>For support: hr.analytics@jayajayamaju.com | Last Updated: June 2025</small>
</div>
""", unsafe_allow_html=True)