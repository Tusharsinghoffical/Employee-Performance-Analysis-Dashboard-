import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from performance_analyzer import EmployeePerformanceAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class PerformanceDashboard:
    def __init__(self):
        self.analyzer = None
        self.load_data()
    
    def load_data(self):
        """Load data and initialize analyzer"""
        try:
            self.analyzer = EmployeePerformanceAnalyzer()
            self.analyzer.load_data('employee_data.csv', 'monthly_performance.csv', 'employee_goals.csv')
            st.success("✅ Data loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Data files not found. Please run data_generator.py first.")
            st.stop()
    
    def main_page(self):
        """Main dashboard page"""
        st.markdown('<h1 class="main-header">📊 Employee Performance Analysis Dashboard</h1>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Employees",
                value=len(self.analyzer.employee_df),
                delta="+5% from last month"
            )
        
        with col2:
            avg_performance = self.analyzer.employee_df['overall_performance_score'].mean()
            st.metric(
                label="Average Performance",
                value=f"{avg_performance:.1f}%",
                delta="+2.3% from last month"
            )
        
        with col3:
            top_performers = len(self.analyzer.employee_df[self.analyzer.employee_df['performance_grade'] == 'A'])
            st.metric(
                label="Top Performers (A)",
                value=top_performers,
                delta=f"{top_performers/len(self.analyzer.employee_df)*100:.1f}% of workforce"
            )
        
        with col4:
            departments = self.analyzer.employee_df['department'].nunique()
            st.metric(
                label="Departments",
                value=departments,
                delta="Active"
            )
        
        # Performance overview
        st.subheader("📈 Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance distribution
            fig = px.histogram(
                self.analyzer.employee_df,
                x='overall_performance_score',
                nbins=20,
                title="Performance Score Distribution",
                labels={'overall_performance_score': 'Performance Score', 'count': 'Number of Employees'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance by department
            dept_performance = self.analyzer.employee_df.groupby('department')['overall_performance_score'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=dept_performance.index,
                y=dept_performance.values,
                title="Average Performance by Department",
                labels={'x': 'Department', 'y': 'Average Performance Score'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics correlation
        st.subheader("🔗 Performance Metrics Correlation")
        score_columns = [col for col in self.analyzer.employee_df.columns if 'score' in col]
        correlation_matrix = self.analyzer.employee_df[score_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Performance Metrics Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def department_analysis_page(self):
        """Department analysis page"""
        st.header("🏢 Department Analysis")
        
        # Department selection
        selected_dept = st.selectbox(
            "Select Department for Detailed Analysis",
            options=['All Departments'] + list(self.analyzer.employee_df['department'].unique())
        )
        
        if selected_dept == 'All Departments':
            dept_data = self.analyzer.employee_df
        else:
            dept_data = self.analyzer.employee_df[self.analyzer.employee_df['department'] == selected_dept]
        
        # Department metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Employees", len(dept_data))
        
        with col2:
            avg_perf = dept_data['overall_performance_score'].mean()
            st.metric("Avg Performance", f"{avg_perf:.1f}%")
        
        with col3:
            avg_tenure = dept_data['tenure_months'].mean()
            st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
        
        # Department visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance grade distribution
            grade_counts = dept_data['performance_grade'].value_counts()
            fig = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                title=f"Performance Grade Distribution - {selected_dept}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tenure vs Performance
            fig = px.scatter(
                dept_data,
                x='tenure_months',
                y='overall_performance_score',
                color='performance_grade',
                title=f"Tenure vs Performance - {selected_dept}",
                labels={'tenure_months': 'Tenure (Months)', 'overall_performance_score': 'Performance Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics by department
        st.subheader("📊 Detailed Performance Metrics")
        
        score_columns = [col for col in dept_data.columns if 'score' in col and col != 'overall_performance_score']
        metrics_data = dept_data[score_columns].mean().reset_index()
        metrics_data.columns = ['Metric', 'Average Score']
        
        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Average Score',
            title=f"Performance Metrics Breakdown - {selected_dept}"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def employee_clustering_page(self):
        """Employee clustering analysis page"""
        st.header("🎯 Employee Clustering Analysis")
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
        
        with col2:
            if st.button("Run Clustering Analysis"):
                with st.spinner("Performing clustering analysis..."):
                    cluster_analysis, cluster_characteristics = self.analyzer.employee_clustering_analysis(n_clusters)
                    
                    # Display cluster analysis results
                    st.subheader("📊 Cluster Analysis Results")
                    
                    # Cluster summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Cluster Summary:**")
                        st.dataframe(cluster_analysis)
                    
                    with col2:
                        st.write("**Cluster Characteristics:**")
                        for cluster, characteristics in cluster_characteristics.items():
                            st.write(f"**{cluster}:**")
                            for key, value in characteristics.items():
                                st.write(f"  {key}: {value}")
                    
                    # Cluster visualization
                    st.subheader("📈 Cluster Visualization")
                    
                    # Prepare data for visualization
                    score_columns = [col for col in self.analyzer.employee_df.columns if 'score' in col]
                    features = self.analyzer.employee_df[score_columns].copy()
                    
                    # PCA for visualization
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    pca = PCA(n_components=2)
                    features_pca = pca.fit_transform(features_scaled)
                    
                    # Create scatter plot
                    cluster_df = pd.DataFrame({
                        'PC1': features_pca[:, 0],
                        'PC2': features_pca[:, 1],
                        'Cluster': self.analyzer.employee_df['cluster'],
                        'Department': self.analyzer.employee_df['department'],
                        'Performance': self.analyzer.employee_df['overall_performance_score']
                    })
                    
                    fig = px.scatter(
                        cluster_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        size='Performance',
                        hover_data=['Department', 'Performance'],
                        title="Employee Clusters (PCA Visualization)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def predictive_analysis_page(self):
        """Predictive analysis page"""
        st.header("🔮 Predictive Performance Analysis")
        
        if st.button("Run Predictive Model"):
            with st.spinner("Training predictive model..."):
                model_results = self.analyzer.performance_prediction_model()
                
                # Model performance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R² Score", f"{model_results['r2_score']:.3f}")
                
                with col2:
                    st.metric("Mean Squared Error", f"{model_results['mse']:.3f}")
                
                with col3:
                    st.metric("Model Accuracy", f"{model_results['r2_score']*100:.1f}%")
                
                # Feature importance
                st.subheader("🎯 Feature Importance")
                
                fig = px.bar(
                    model_results['feature_importance'],
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance for Performance Prediction"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction interface
                st.subheader("📊 Performance Prediction Tool")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Enter employee metrics for prediction:**")
                    
                    productivity = st.slider("Productivity Score", 0, 100, 75)
                    quality = st.slider("Quality Score", 0, 100, 80)
                    attendance = st.slider("Attendance Rate", 0, 100, 92)
                    teamwork = st.slider("Teamwork Score", 0, 100, 78)
                    initiative = st.slider("Initiative Score", 0, 100, 70)
                
                with col2:
                    communication = st.slider("Communication Score", 0, 100, 75)
                    problem_solving = st.slider("Problem Solving Score", 0, 100, 72)
                    adaptability = st.slider("Adaptability Score", 0, 100, 76)
                    leadership = st.slider("Leadership Score", 0, 100, 65)
                    tenure = st.slider("Tenure (Months)", 0, 120, 24)
                    projects = st.slider("Projects Completed", 0, 20, 8)
                    training = st.slider("Training Hours", 0, 100, 40)
                
                if st.button("Predict Performance"):
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'productivity_score': [productivity],
                        'quality_score': [quality],
                        'attendance_rate': [attendance],
                        'teamwork_score': [teamwork],
                        'initiative_score': [initiative],
                        'communication_score': [communication],
                        'problem_solving_score': [problem_solving],
                        'adaptability_score': [adaptability],
                        'leadership_score': [leadership],
                        'tenure_months': [tenure],
                        'projects_completed': [projects],
                        'training_hours': [training]
                    })
                    
                    # Make prediction
                    prediction = model_results['model'].predict(input_data)[0]
                    
                    st.success(f"🎯 Predicted Performance Score: {prediction:.1f}%")
                    
                    # Performance grade
                    if prediction >= 90:
                        grade = "A"
                        color = "green"
                    elif prediction >= 80:
                        grade = "B"
                        color = "blue"
                    elif prediction >= 70:
                        grade = "C"
                        color = "orange"
                    elif prediction >= 60:
                        grade = "D"
                        color = "red"
                    else:
                        grade = "F"
                        color = "darkred"
                    
                    st.markdown(f"**Performance Grade: <span style='color: {color}; font-size: 24px;'>{grade}</span>**", unsafe_allow_html=True)
    
    def trends_analysis_page(self):
        """Trends analysis page"""
        st.header("📈 Performance Trends Analysis")
        
        if self.analyzer.monthly_df is None:
            st.warning("Monthly data not available for trend analysis.")
            return
        
        # Time period selection
        col1, col2 = st.columns(2)
        
        with col1:
            months = sorted(self.analyzer.monthly_df['month'].unique())
            start_month = st.selectbox("Start Month", months, index=0)
        
        with col2:
            end_month = st.selectbox("End Month", months, index=len(months)-1)
        
        # Filter data
        mask = (self.analyzer.monthly_df['month'] >= start_month) & (self.analyzer.monthly_df['month'] <= end_month)
        filtered_data = self.analyzer.monthly_df[mask]
        
        # Overall performance trend
        monthly_avg = filtered_data.groupby('month')['overall_performance_score'].mean()
        
        fig = px.line(
            x=monthly_avg.index,
            y=monthly_avg.values,
            title="Overall Performance Trend",
            labels={'x': 'Month', 'y': 'Average Performance Score'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Department trends
        dept_trends = filtered_data.groupby(['month', 'department'])['overall_performance_score'].mean().unstack()
        
        fig = px.line(
            dept_trends,
            title="Performance Trends by Department"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance heatmap
        st.subheader("🔥 Performance Heatmap")
        
        heatmap_data = filtered_data.pivot_table(
            values='overall_performance_score',
            index='month',
            columns='department',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            heatmap_data,
            title="Performance Heatmap by Department and Month",
            color_continuous_scale='YlOrRd'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def goals_analysis_page(self):
        """Goals analysis page"""
        st.header("🎯 Employee Goals Analysis")
        
        if self.analyzer.goals_df is None:
            st.warning("Goals data not available.")
            return
        
        # Goals overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Goals", len(self.analyzer.goals_df))
        
        with col2:
            completed_goals = len(self.analyzer.goals_df[self.analyzer.goals_df['status'] == 'Completed'])
            st.metric("Completed Goals", completed_goals)
        
        with col3:
            avg_achievement = self.analyzer.goals_df['achievement_rate'].mean()
            st.metric("Avg Achievement Rate", f"{avg_achievement:.1f}%")
        
        with col4:
            goal_types = self.analyzer.goals_df['goal_type'].nunique()
            st.metric("Goal Types", goal_types)
        
        # Goals visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Achievement rate distribution
            fig = px.histogram(
                self.analyzer.goals_df,
                x='achievement_rate',
                nbins=20,
                title="Goal Achievement Rate Distribution",
                labels={'achievement_rate': 'Achievement Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Goals by type
            goal_type_counts = self.analyzer.goals_df['goal_type'].value_counts()
            fig = px.pie(
                values=goal_type_counts.values,
                names=goal_type_counts.index,
                title="Goals by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Department goals analysis
        st.subheader("🏢 Goals by Department")
        
        dept_achievement = self.analyzer.goals_df.groupby('department')['achievement_rate'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=dept_achievement.index,
            y=dept_achievement.values,
            title="Average Achievement Rate by Department",
            labels={'x': 'Department', 'y': 'Average Achievement Rate (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Goals status
        st.subheader("📊 Goals Status")
        
        status_counts = self.analyzer.goals_df['status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Goals Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application"""
    dashboard = PerformanceDashboard()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Main Dashboard", "🏢 Department Analysis", "🎯 Employee Clustering", 
         "🔮 Predictive Analysis", "📈 Trends Analysis", "🎯 Goals Analysis"]
    )
    
    # Page routing
    if page == "🏠 Main Dashboard":
        dashboard.main_page()
    elif page == "🏢 Department Analysis":
        dashboard.department_analysis_page()
    elif page == "🎯 Employee Clustering":
        dashboard.employee_clustering_page()
    elif page == "🔮 Predictive Analysis":
        dashboard.predictive_analysis_page()
    elif page == "📈 Trends Analysis":
        dashboard.trends_analysis_page()
    elif page == "🎯 Goals Analysis":
        dashboard.goals_analysis_page()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.markdown(f"**Employees:** {len(dashboard.analyzer.employee_df)}")
    st.sidebar.markdown(f"**Departments:** {dashboard.analyzer.employee_df['department'].nunique()}")
    st.sidebar.markdown(f"**Avg Performance:** {dashboard.analyzer.employee_df['overall_performance_score'].mean():.1f}%")
    
    if dashboard.analyzer.monthly_df is not None:
        st.sidebar.markdown(f"**Monthly Records:** {len(dashboard.analyzer.monthly_df)}")
    
    if dashboard.analyzer.goals_df is not None:
        st.sidebar.markdown(f"**Goals:** {len(dashboard.analyzer.goals_df)}")

if __name__ == "__main__":
    main() 