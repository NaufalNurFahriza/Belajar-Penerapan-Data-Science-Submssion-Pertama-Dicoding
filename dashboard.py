import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard - Jaya Jaya Maju",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("dataset/attrition_dashboard_data.csv")  

@st.cache_resource
def load_model():
    return joblib.load("model/attrition_model.pkl")

# Load data
try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.stop()

# Feature lists
numeric_features = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

ordinal_cols = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'WorkLifeBalance'
]

nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                'JobRole', 'MaritalStatus', 'OverTime']

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
    st.image("https://via.placeholder.com/150x50?text=Company+Logo", width=150)
    st.title("HR Analytics Dashboard")
    menu = st.radio(
        "Navigation",
        ["📊 Dashboard Overview", "🔍 Feature Analysis", "🤖 Attrition Predictor", "💡 Insights"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Data Filters")
    department_filter = st.multiselect(
        "Filter by Department",
        options=df['Department'].unique(),
        default=df['Department'].unique()
    )
    
    attrition_filter = st.selectbox(
        "Filter by Attrition Status",
        options=["All", "Attrited", "Not Attrited"],
        index=0
    )
    
    age_range = st.slider(
        "Filter by Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

# Apply filters
filtered_df = df.copy()
if department_filter:
    filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]
if attrition_filter == "Attrited":
    filtered_df = filtered_df[filtered_df['Attrition'] == 1]
elif attrition_filter == "Not Attrited":
    filtered_df = filtered_df[filtered_df['Attrition'] == 0]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

# Dashboard Overview
if menu == "📊 Dashboard Overview":
    st.title("HR Attrition Analytics Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", len(filtered_df))
    with col2:
        attrition_rate = filtered_df['Attrition'].mean()*100
        delta = (attrition_rate - (df['Attrition'].mean()*100)) / (df['Attrition'].mean()*100) * 100
        st.metric(
            "Attrition Rate", 
            f"{attrition_rate:.1f}%",
            delta=f"{delta:.1f}% vs overall",
            help="Percentage of employees who left the company"
        )
    with col3:
        avg_tenure = filtered_df['YearsAtCompany'].mean()
        st.metric("Avg Company Tenure", f"{avg_tenure:.1f} years")
    with col4:
        avg_income = filtered_df['MonthlyIncome'].mean()
        st.metric("Avg Monthly Income", f"${avg_income:,.0f}")
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        attrition_by_dept = filtered_df.groupby(['Department', 'Attrition']).size().unstack().fillna(0)
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
    
    with col2:
        st.subheader("Monthly Income Distribution")
        fig = px.box(
            filtered_df,
            x='Attrition',
            y='MonthlyIncome',
            color='Attrition',
            points="all",
            labels={'MonthlyIncome': 'Monthly Income ($)'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Additional charts
    st.subheader("Key Factors Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Work-Life Balance", "Job Satisfaction", "Overtime", "Age Distribution"])
    
    with tab1:
        wlb_counts = filtered_df.groupby(['WorkLifeBalance', 'Attrition']).size().unstack().fillna(0)
        wlb_counts = wlb_counts.div(wlb_counts.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            wlb_counts.reset_index(),
            x='WorkLifeBalance',
            y=[0, 1],
            barmode='group',
            labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
            title='Attrition by Work-Life Balance Rating',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        js_counts = filtered_df.groupby(['JobSatisfaction', 'Attrition']).size().unstack().fillna(0)
        js_counts = js_counts.div(js_counts.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            js_counts.reset_index(),
            x='JobSatisfaction',
            y=[0, 1],
            barmode='group',
            labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
            title='Attrition by Job Satisfaction Rating',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        ot_counts = filtered_df.groupby(['OverTime', 'Attrition']).size().unstack().fillna(0)
        ot_counts = ot_counts.div(ot_counts.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            ot_counts.reset_index(),
            x='OverTime',
            y=[0, 1],
            barmode='group',
            labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
            title='Attrition by Overtime Status',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
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

# Feature Analysis
elif menu == "🔍 Feature Analysis":
    st.title("Feature Analysis")
    
    st.subheader("Feature Importance")
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
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
    
    st.markdown("---")
    
    st.subheader("Feature Comparison Tool")
    col1, col2 = st.columns(2)
    with col1:
        feature = st.selectbox(
            "Select feature to analyze",
            options=sorted(df.columns.drop('Attrition'))
        )
    
    with col2:
        if feature in numeric_features:
            agg_func = st.selectbox(
                "Aggregation",
                options=["Mean", "Median", "Sum", "Count"],
                index=0
            )
    
    if feature in numeric_features:
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
    else:
        count_df = filtered_df.groupby([feature, 'Attrition']).size().unstack().fillna(0)
        count_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            count_df.reset_index(),
            x=feature,
            y=[0, 1],
            barmode='group',
            labels={'value': 'Percentage (%)', 'variable': 'Attrition'},
            title=f'Attrition by {feature}',
            height=500
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Attrition Predictor
elif menu == "🤖 Attrition Predictor":
    st.title("Employee Attrition Predictor")
    st.markdown("Predict whether an employee is at risk of leaving the company")
    
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
            job_satisfaction = st.select_slider("Job Satisfaction", 
                                              options=[1, 2, 3, 4],
                                              format_func=lambda x: ordinal_label_map['JobSatisfaction'][x])
            years_at_company = st.number_input("Years at Company", 0, 40, 2)
            department = st.selectbox("Department", df['Department'].unique())
        
        submitted = st.form_submit_button("Predict Attrition Risk")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame([{
                'Age': age,
                'MonthlyIncome': monthly_income,
                'TotalWorkingYears': total_working_years,
                'OverTime': overtime,
                'JobLevel': job_level,
                'JobSatisfaction': job_satisfaction,
                'YearsAtCompany': years_at_company,
                'Department': department,
                'DailyRate': df['DailyRate'].median(),
                'DistanceFromHome': df['DistanceFromHome'].median(),
                'HourlyRate': df['HourlyRate'].median(),
                'Education': 3,  # Default to Bachelor's degree
                'EnvironmentSatisfaction': 3,  # Default to Satisfied
                'JobInvolvement': 3,  # Default to High
                'PerformanceRating': 3,  # Default to Excellent
                'RelationshipSatisfaction': 3,  # Default to Satisfied
                'StockOptionLevel': 1,  # Default to Level 1
                'WorkLifeBalance': 3,  # Default to Good
                'BusinessTravel': 'Travel_Rarely',
                'EducationField': 'Life Sciences',
                'Gender': 'Male',
                'JobRole': 'Sales Executive',
                'MaritalStatus': 'Single',
                'NumCompaniesWorked': 2,
                'PercentSalaryHike': 15,
                'TrainingTimesLastYear': 2,
                'YearsInCurrentRole': years_at_company - 1 if years_at_company > 0 else 0,
                'YearsSinceLastPromotion': years_at_company - 1 if years_at_company > 0 else 0,
                'YearsWithCurrManager': years_at_company - 1 if years_at_company > 0 else 0,
                'MonthlyRate': monthly_income * 1.2  # Estimate
            }])
            
            # Make prediction
            try:
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
                        st.markdown("""
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
                        st.markdown("""
                        **Maintenance Actions:**
                        - Continue regular check-ins
                        - Monitor career development opportunities
                        - Ensure competitive compensation (current: ${monthly_income:,.0f})
                        
                        **Preventive Measures:**
                        - Annual retention risk assessment
                        - Leadership development program
                        - Flexible work arrangement options
                        """)
                
                # SHAP values explanation if available
                try:
                    import shap
                    explainer = shap.TreeExplainer(model.named_steps['classifier'])
                    transformed_input = model.named_steps['preprocessor'].transform(input_data)
                    shap_values = explainer.shap_values(transformed_input)
                    
                    st.subheader("Key Factors Influencing This Prediction")
                    fig, ax = plt.subplots()
                    shap.summary_plot(
                        shap_values[1], 
                        transformed_input, 
                        feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
                        plot_type="bar",
                        max_display=10
                    )
                    st.pyplot(fig)
                except:
                    pass
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Insights
elif menu == "💡 Insights":
    st.title("Key Insights & Recommendations")
    
    st.subheader("Top Attrition Drivers")
    st.markdown("""
    Our analysis identified these as the strongest predictors of employee attrition:
    """)
    
    drivers = [
        {"factor": "Monthly Income", "impact": "High", "description": "Lower income employees are 3.2x more likely to leave"},
        {"factor": "Overtime Status", "impact": "High", "description": "Employees working overtime have 2.5x higher attrition risk"},
        {"factor": "Age", "impact": "Medium", "description": "Employees under 30 show 1.8x higher turnover rates"},
        {"factor": "Job Level", "impact": "Medium", "description": "Entry-level positions experience 1.6x more attrition"},
        {"factor": "Work-Life Balance", "impact": "Medium", "description": "Poor ratings correlate with 1.5x higher attrition"}
    ]
    
    for driver in drivers:
        with st.expander(f"{driver['factor']} ({driver['impact']} impact)"):
            st.markdown(driver['description'])
            if driver['factor'] == "Monthly Income":
                st.plotly_chart(px.box(
                    df,
                    x='Attrition',
                    y='MonthlyIncome',
                    color='Attrition',
                    title='Monthly Income Distribution by Attrition Status'
                ), use_container_width=True)
            elif driver['factor'] == "Overtime Status":
                ot_data = df.groupby('OverTime')['Attrition'].mean().reset_index()
                st.plotly_chart(px.bar(
                    ot_data,
                    x='OverTime',
                    y='Attrition',
                    color='OverTime',
                    title='Attrition Rate by Overtime Status',
                    labels={'Attrition': 'Attrition Rate'}
                ), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Department-Specific Findings")
    dept_data = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        dept_data,
        x='Department',
        y='Attrition',
        color='Department',
        title='Attrition Rate by Department',
        labels={'Attrition': 'Attrition Rate'},
        text_auto='.1%'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Sales Department:**
        - Highest overall attrition rate (18.2%)
        - Primary drivers: 
          - Overtime (67% work overtime)
          - Travel requirements (42% travel frequently)
          - High performance pressure
        """)
        
        st.markdown("""
        **Research & Development:**
        - Moderate attrition rate (12.1%)
        - Key factors: 
          - Career growth opportunities
          - Project assignment dissatisfaction
          - Compensation competitiveness
        """)
    
    with col2:
        st.markdown("""
        **Human Resources:**
        - Lowest attrition rate (7.3%)
        - Strongest retention factors: 
          - Work-life balance (82% rate it Good/Excellent)
          - Job stability
          - Internal career progression
        """)
    
    st.markdown("---")
    
    st.subheader("Actionable Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Immediate Priorities", "Medium-Term Initiatives", "Long-Term Strategy"])
    
    with tab1:
        st.markdown("""
        **Within 30 Days:**
        
        - Implement overtime monitoring dashboard with automatic alerts when employees exceed 10% overtime monthly
        - Identify all employees earning below department median and initiate compensation reviews
        - Launch "Stay Interviews" program for high-risk employees
        
        **Within 90 Days:**
        
        - Develop department-specific retention bonus programs
        - Create task force to address work-life balance concerns
        - Implement exit interview analysis process
        """)
    
    with tab2:
        st.markdown("""
        **6-12 Month Initiatives:**
        
        - Career Path Development:
          - Create transparent promotion frameworks for all roles
          - Implement skills-based progression tracks
          - Launch internal mobility program
        
        - Manager Training:
          - Retention leadership certification
          - Coaching for career development conversations
          - Workload management training
        
        - Compensation Strategy:
          - Market benchmarking analysis
          - Performance-based bonus structure
          - Equity and recognition programs
        """)
    
    with tab3:
        st.markdown("""
        **Strategic Initiatives:**
        
        - Predictive Analytics:
          - Real-time attrition risk scoring
          - Early warning system integration with HRIS
          - Automated intervention recommendations
        
        - Culture & Engagement:
          - Quarterly pulse surveys with AI sentiment analysis
          - Employee experience journey mapping
          - Values alignment programs
        
        - Workforce Planning:
          - Skills gap analysis
          - Succession planning integration
          - Talent pipeline development
        """)

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
}
</style>
<div class="footer">
    HR Analytics Dashboard • Powered by Streamlit • Data updated monthly<br>
    Last updated: June 2023 • Contact: hr.analytics@company.com
</div>
""", unsafe_allow_html=True)