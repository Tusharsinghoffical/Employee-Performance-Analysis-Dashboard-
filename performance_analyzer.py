import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EmployeePerformanceAnalyzer:
    def __init__(self, employee_data_path=None, monthly_data_path=None, goals_data_path=None):
        """Initialize the analyzer with data paths"""
        self.employee_df = None
        self.monthly_df = None
        self.goals_df = None
        
        if employee_data_path:
            self.load_data(employee_data_path, monthly_data_path, goals_data_path)
    
    def load_data(self, employee_data_path, monthly_data_path=None, goals_data_path=None):
        """Load employee performance data"""
        self.employee_df = pd.read_csv(employee_data_path)
        
        if monthly_data_path:
            self.monthly_df = pd.read_csv(monthly_data_path)
        
        if goals_data_path:
            self.goals_df = pd.read_csv(goals_data_path)
        
        print(f"Loaded {len(self.employee_df)} employee records")
        if self.monthly_df is not None:
            print(f"Loaded {len(self.monthly_df)} monthly performance records")
        if self.goals_df is not None:
            print(f"Loaded {len(self.goals_df)} goal records")
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        if self.employee_df is None:
            print("No employee data loaded")
            return
        
        # Performance metrics summary
        score_columns = [col for col in self.employee_df.columns if 'score' in col]
        
        stats_summary = {
            'Total Employees': len(self.employee_df),
            'Departments': self.employee_df['department'].nunique(),
            'Average Overall Performance': self.employee_df['overall_performance_score'].mean(),
            'Performance Std Dev': self.employee_df['overall_performance_score'].std(),
            'Top Performers (A Grade)': len(self.employee_df[self.employee_df['performance_grade'] == 'A']),
            'Low Performers (F Grade)': len(self.employee_df[self.employee_df['performance_grade'] == 'F'])
        }
        
        # Department-wise statistics
        dept_stats = self.employee_df.groupby('department').agg({
            'overall_performance_score': ['mean', 'std', 'count'],
            'tenure_months': 'mean',
            'projects_completed': 'mean'
        }).round(2)
        
        return stats_summary, dept_stats
    
    def performance_distribution_analysis(self):
        """Analyze performance distributions"""
        if self.employee_df is None:
            return
        
        score_columns = [col for col in self.employee_df.columns if 'score' in col]
        
        # Create distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(score_columns):
            if i < 9:  # Limit to 9 plots
                axes[i].hist(self.employee_df[metric], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(self.employee_df[metric].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.employee_df[metric].mean():.1f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        correlation_matrix = self.employee_df[score_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Performance Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def department_performance_analysis(self):
        """Analyze performance by department"""
        if self.employee_df is None:
            return
        
        # Department performance comparison
        dept_performance = self.employee_df.groupby('department').agg({
            'overall_performance_score': ['mean', 'std', 'count'],
            'productivity_score': 'mean',
            'quality_score': 'mean',
            'teamwork_score': 'mean',
            'leadership_score': 'mean'
        }).round(2)
        
        # Visualize department performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall performance by department
        dept_means = self.employee_df.groupby('department')['overall_performance_score'].mean().sort_values(ascending=False)
        axes[0, 0].bar(dept_means.index, dept_means.values, color='skyblue')
        axes[0, 0].set_title('Average Performance by Department')
        axes[0, 0].set_ylabel('Average Performance Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Employee count by department
        dept_counts = self.employee_df['department'].value_counts()
        axes[0, 1].pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Employee Distribution by Department')
        
        # Performance grade distribution by department
        grade_dept = pd.crosstab(self.employee_df['department'], self.employee_df['performance_grade'])
        grade_dept.plot(kind='bar', ax=axes[1, 0], stacked=True)
        axes[1, 0].set_title('Performance Grade Distribution by Department')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Grade')
        
        # Tenure vs Performance scatter
        axes[1, 1].scatter(self.employee_df['tenure_months'], self.employee_df['overall_performance_score'], alpha=0.6)
        axes[1, 1].set_xlabel('Tenure (Months)')
        axes[1, 1].set_ylabel('Overall Performance Score')
        axes[1, 1].set_title('Tenure vs Performance')
        
        plt.tight_layout()
        plt.show()
        
        return dept_performance
    
    def employee_clustering_analysis(self, n_clusters=4):
        """Perform clustering analysis to identify employee segments"""
        if self.employee_df is None:
            return
        
        # Prepare features for clustering
        score_columns = [col for col in self.employee_df.columns if 'score' in col]
        features = self.employee_df[score_columns].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to dataframe
        self.employee_df['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = self.employee_df.groupby('cluster').agg({
            'overall_performance_score': ['mean', 'std', 'count'],
            'tenure_months': 'mean',
            'projects_completed': 'mean',
            'department': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        # Visualize clusters
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Employee Clusters (PCA Visualization)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
        
        # Cluster characteristics
        cluster_characteristics = {}
        for cluster_id in range(n_clusters):
            cluster_data = self.employee_df[self.employee_df['cluster'] == cluster_id]
            cluster_characteristics[f'Cluster {cluster_id}'] = {
                'Size': len(cluster_data),
                'Avg Performance': cluster_data['overall_performance_score'].mean(),
                'Top Departments': cluster_data['department'].value_counts().head(2).to_dict(),
                'Performance Range': f"{cluster_data['overall_performance_score'].min():.1f} - {cluster_data['overall_performance_score'].max():.1f}"
            }
        
        return cluster_analysis, cluster_characteristics
    
    def performance_prediction_model(self):
        """Build a predictive model for performance"""
        if self.employee_df is None:
            return
        
        # Prepare features
        feature_columns = [col for col in self.employee_df.columns if 'score' in col and col != 'overall_performance_score']
        feature_columns.extend(['tenure_months', 'projects_completed', 'training_hours'])
        
        X = self.employee_df[feature_columns].copy()
        y = self.employee_df['overall_performance_score']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance for Performance Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'model': model,
            'mse': mse,
            'r2_score': r2,
            'feature_importance': feature_importance
        }
    
    def trend_analysis(self):
        """Analyze performance trends over time"""
        if self.monthly_df is None:
            print("Monthly data not available for trend analysis")
            return
        
        # Monthly performance trends
        monthly_trends = self.monthly_df.groupby('month').agg({
            'overall_performance_score': ['mean', 'std'],
            'productivity_score': 'mean',
            'quality_score': 'mean',
            'teamwork_score': 'mean'
        }).round(2)
        
        # Visualize trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall performance trend
        monthly_avg = self.monthly_df.groupby('month')['overall_performance_score'].mean()
        axes[0, 0].plot(monthly_avg.index, monthly_avg.values, marker='o')
        axes[0, 0].set_title('Overall Performance Trend')
        axes[0, 0].set_ylabel('Average Performance Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Department trends
        dept_trends = self.monthly_df.groupby(['month', 'department'])['overall_performance_score'].mean().unstack()
        dept_trends.plot(ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('Performance Trends by Department')
        axes[0, 1].set_ylabel('Average Performance Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Performance distribution over time
        monthly_df_pivot = self.monthly_df.pivot_table(
            values='overall_performance_score', 
            index='month', 
            columns='department', 
            aggfunc='mean'
        )
        sns.heatmap(monthly_df_pivot, ax=axes[1, 0], cmap='YlOrRd', annot=True, fmt='.1f')
        axes[1, 0].set_title('Performance Heatmap by Department and Month')
        
        # Performance volatility
        monthly_volatility = self.monthly_df.groupby('month')['overall_performance_score'].std()
        axes[1, 1].plot(monthly_volatility.index, monthly_volatility.values, marker='s', color='red')
        axes[1, 1].set_title('Performance Volatility Over Time')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return monthly_trends
    
    def goals_analysis(self):
        """Analyze employee goals and achievements"""
        if self.goals_df is None:
            print("Goals data not available")
            return
        
        # Goals analysis
        goals_summary = {
            'Total Goals': len(self.goals_df),
            'Completed Goals': len(self.goals_df[self.goals_df['status'] == 'Completed']),
            'In Progress Goals': len(self.goals_df[self.goals_df['status'] == 'In Progress']),
            'Average Achievement Rate': self.goals_df['achievement_rate'].mean(),
            'Goals by Type': self.goals_df['goal_type'].value_counts().to_dict()
        }
        
        # Visualize goals
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Achievement rate distribution
        axes[0, 0].hist(self.goals_df['achievement_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Goal Achievement Rate Distribution')
        axes[0, 0].set_xlabel('Achievement Rate (%)')
        axes[0, 0].set_ylabel('Number of Goals')
        
        # Goals by type
        goal_types = self.goals_df['goal_type'].value_counts()
        axes[0, 1].pie(goal_types.values, labels=goal_types.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Goals by Type')
        
        # Achievement rate by department
        dept_achievement = self.goals_df.groupby('department')['achievement_rate'].mean().sort_values(ascending=False)
        axes[1, 0].bar(dept_achievement.index, dept_achievement.values, color='lightgreen')
        axes[1, 0].set_title('Average Achievement Rate by Department')
        axes[1, 0].set_ylabel('Average Achievement Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Goals status
        status_counts = self.goals_df['status'].value_counts()
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Goals Status Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return goals_summary
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive performance analysis report"""
        if self.employee_df is None:
            print("No data loaded for analysis")
            return
        
        print("=" * 60)
        print("EMPLOYEE PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        
        # Basic statistics
        stats_summary, dept_stats = self.basic_statistics()
        print("\n1. BASIC STATISTICS:")
        for key, value in stats_summary.items():
            print(f"   {key}: {value}")
        
        print("\n2. DEPARTMENT PERFORMANCE:")
        print(dept_stats)
        
        # Performance distribution
        print("\n3. PERFORMANCE DISTRIBUTION ANALYSIS:")
        correlation_matrix = self.performance_distribution_analysis()
        
        # Department analysis
        print("\n4. DEPARTMENT PERFORMANCE ANALYSIS:")
        dept_performance = self.department_performance_analysis()
        
        # Clustering analysis
        print("\n5. EMPLOYEE CLUSTERING ANALYSIS:")
        cluster_analysis, cluster_characteristics = self.employee_clustering_analysis()
        print("Cluster Analysis Results:")
        print(cluster_analysis)
        
        print("\nCluster Characteristics:")
        for cluster, characteristics in cluster_characteristics.items():
            print(f"\n{cluster}:")
            for key, value in characteristics.items():
                print(f"  {key}: {value}")
        
        # Predictive modeling
        print("\n6. PERFORMANCE PREDICTION MODEL:")
        model_results = self.performance_prediction_model()
        print(f"Model R² Score: {model_results['r2_score']:.3f}")
        print(f"Mean Squared Error: {model_results['mse']:.3f}")
        
        # Trend analysis
        if self.monthly_df is not None:
            print("\n7. TREND ANALYSIS:")
            monthly_trends = self.trend_analysis()
        
        # Goals analysis
        if self.goals_df is not None:
            print("\n8. GOALS ANALYSIS:")
            goals_summary = self.goals_analysis()
            for key, value in goals_summary.items():
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("REPORT COMPLETED")
        print("=" * 60)

if __name__ == "__main__":
    # Example usage
    analyzer = EmployeePerformanceAnalyzer()
    
    # Load data if available
    try:
        analyzer.load_data('employee_data.csv', 'monthly_performance.csv', 'employee_goals.csv')
        analyzer.generate_comprehensive_report()
    except FileNotFoundError:
        print("Data files not found. Please run data_generator.py first to generate sample data.") 