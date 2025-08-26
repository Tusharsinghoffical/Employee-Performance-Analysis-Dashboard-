import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xlsxwriter
from performance_analyzer import EmployeePerformanceAnalyzer
import warnings
warnings.filterwarnings('ignore')

class PerformanceReportGenerator:
    def __init__(self, analyzer):
        """Initialize report generator with analyzer instance"""
        self.analyzer = analyzer
        self.report_data = {}
    
    def generate_excel_report(self, filename="employee_performance_report.xlsx"):
        """Generate comprehensive Excel report with multiple sheets"""
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'fg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            # Generate all report sections
            self._generate_executive_summary(writer, header_format, title_format)
            self._generate_department_analysis(writer, header_format, title_format)
            self._generate_performance_metrics(writer, header_format, title_format)
            self._generate_employee_clustering(writer, header_format, title_format)
            self._generate_predictive_analysis(writer, header_format, title_format)
            self._generate_trends_analysis(writer, header_format, title_format)
            self._generate_goals_analysis(writer, header_format, title_format)
            self._generate_recommendations(writer, header_format, title_format)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Report generated successfully: {filename}")
        return filename
    
    def _generate_executive_summary(self, writer, header_format, title_format):
        """Generate executive summary sheet"""
        worksheet = writer.book.add_worksheet('Executive Summary')
        
        # Title
        worksheet.merge_range('A1:D1', 'EMPLOYEE PERFORMANCE ANALYSIS - EXECUTIVE SUMMARY', title_format)
        
        # Basic statistics
        stats_summary, dept_stats = self.analyzer.basic_statistics()
        
        # Key metrics
        worksheet.write('A3', 'KEY METRICS', header_format)
        worksheet.write('A4', 'Metric')
        worksheet.write('B4', 'Value')
        worksheet.write('C4', 'Description')
        
        row = 5
        for metric, value in stats_summary.items():
            worksheet.write(f'A{row}', metric)
            worksheet.write(f'B{row}', value)
            if metric == 'Total Employees':
                worksheet.write(f'C{row}', 'Total number of employees in the organization')
            elif metric == 'Departments':
                worksheet.write(f'C{row}', 'Number of different departments')
            elif metric == 'Average Overall Performance':
                worksheet.write(f'C{row}', 'Average performance score across all employees')
            elif metric == 'Performance Std Dev':
                worksheet.write(f'C{row}', 'Standard deviation of performance scores')
            elif metric == 'Top Performers (A Grade)':
                worksheet.write(f'C{row}', 'Number of employees with A grade performance')
            elif metric == 'Low Performers (F Grade)':
                worksheet.write(f'C{row}', 'Number of employees with F grade performance')
            row += 1
        
        # Department performance summary
        worksheet.write(f'A{row+2}', 'DEPARTMENT PERFORMANCE SUMMARY', header_format)
        dept_stats.to_excel(writer, sheet_name='Executive Summary', startrow=row+3, startcol=0)
        
        # Performance distribution
        worksheet.write(f'A{row+5}', 'PERFORMANCE GRADE DISTRIBUTION', header_format)
        grade_dist = self.analyzer.employee_df['performance_grade'].value_counts()
        worksheet.write(f'A{row+6}', 'Grade')
        worksheet.write(f'B{row+6}', 'Count')
        worksheet.write(f'C{row+6}', 'Percentage')
        
        row += 7
        for grade, count in grade_dist.items():
            percentage = (count / len(self.analyzer.employee_df)) * 100
            worksheet.write(f'A{row}', grade)
            worksheet.write(f'B{row}', count)
            worksheet.write(f'C{row}', f"{percentage:.1f}%")
            row += 1
    
    def _generate_department_analysis(self, writer, header_format, title_format):
        """Generate department analysis sheet"""
        worksheet = writer.book.add_worksheet('Department Analysis')
        
        # Title
        worksheet.merge_range('A1:E1', 'DEPARTMENT PERFORMANCE ANALYSIS', title_format)
        
        # Department performance comparison
        dept_performance = self.analyzer.employee_df.groupby('department').agg({
            'overall_performance_score': ['mean', 'std', 'count'],
            'productivity_score': 'mean',
            'quality_score': 'mean',
            'teamwork_score': 'mean',
            'leadership_score': 'mean',
            'tenure_months': 'mean',
            'projects_completed': 'mean'
        }).round(2)
        
        # Flatten column names
        dept_performance.columns = ['_'.join(col).strip() for col in dept_performance.columns]
        dept_performance.reset_index(inplace=True)
        
        # Write department performance data
        worksheet.write('A3', 'DEPARTMENT PERFORMANCE METRICS', header_format)
        dept_performance.to_excel(writer, sheet_name='Department Analysis', startrow=4, startcol=0, index=False)
        
        # Department rankings
        worksheet.write('A15', 'DEPARTMENT RANKINGS', header_format)
        worksheet.write('A16', 'Rank')
        worksheet.write('B16', 'Department')
        worksheet.write('C16', 'Average Performance')
        worksheet.write('D16', 'Employee Count')
        
        dept_rankings = self.analyzer.employee_df.groupby('department')['overall_performance_score'].mean().sort_values(ascending=False)
        
        row = 17
        for rank, (dept, score) in enumerate(dept_rankings.items(), 1):
            count = len(self.analyzer.employee_df[self.analyzer.employee_df['department'] == dept])
            worksheet.write(f'A{row}', rank)
            worksheet.write(f'B{row}', dept)
            worksheet.write(f'C{row}', f"{score:.1f}")
            worksheet.write(f'D{row}', count)
            row += 1
    
    def _generate_performance_metrics(self, writer, header_format, title_format):
        """Generate performance metrics analysis sheet"""
        worksheet = writer.book.add_worksheet('Performance Metrics')
        
        # Title
        worksheet.merge_range('A1:F1', 'DETAILED PERFORMANCE METRICS ANALYSIS', title_format)
        
        # Performance metrics summary
        score_columns = [col for col in self.analyzer.employee_df.columns if 'score' in col]
        
        worksheet.write('A3', 'PERFORMANCE METRICS SUMMARY', header_format)
        worksheet.write('A4', 'Metric')
        worksheet.write('B4', 'Mean')
        worksheet.write('C4', 'Std Dev')
        worksheet.write('D4', 'Min')
        worksheet.write('E4', 'Max')
        worksheet.write('F4', 'Median')
        
        row = 5
        for metric in score_columns:
            data = self.analyzer.employee_df[metric]
            worksheet.write(f'A{row}', metric.replace('_', ' ').title())
            worksheet.write(f'B{row}', f"{data.mean():.2f}")
            worksheet.write(f'C{row}', f"{data.std():.2f}")
            worksheet.write(f'D{row}', f"{data.min():.2f}")
            worksheet.write(f'E{row}', f"{data.max():.2f}")
            worksheet.write(f'F{row}', f"{data.median():.2f}")
            row += 1
        
        # Correlation matrix
        worksheet.write(f'A{row+2}', 'PERFORMANCE METRICS CORRELATION MATRIX', header_format)
        correlation_matrix = self.analyzer.employee_df[score_columns].corr()
        correlation_matrix.to_excel(writer, sheet_name='Performance Metrics', startrow=row+3, startcol=0)
        
        # Top and bottom performers
        worksheet.write(f'A{row+15}', 'TOP PERFORMERS (A Grade)', header_format)
        top_performers = self.analyzer.employee_df[self.analyzer.employee_df['performance_grade'] == 'A']
        top_performers[['employee_id', 'department', 'position', 'overall_performance_score']].to_excel(
            writer, sheet_name='Performance Metrics', startrow=row+16, startcol=0, index=False
        )
        
        worksheet.write(f'A{row+25}', 'LOW PERFORMERS (F Grade)', header_format)
        low_performers = self.analyzer.employee_df[self.analyzer.employee_df['performance_grade'] == 'F']
        low_performers[['employee_id', 'department', 'position', 'overall_performance_score']].to_excel(
            writer, sheet_name='Performance Metrics', startrow=row+26, startcol=0, index=False
        )
    
    def _generate_employee_clustering(self, writer, header_format, title_format):
        """Generate employee clustering analysis sheet"""
        worksheet = writer.book.add_worksheet('Employee Clustering')
        
        # Title
        worksheet.merge_range('A1:D1', 'EMPLOYEE CLUSTERING ANALYSIS', title_format)
        
        # Run clustering analysis
        cluster_analysis, cluster_characteristics = self.analyzer.employee_clustering_analysis()
        
        worksheet.write('A3', 'CLUSTER ANALYSIS RESULTS', header_format)
        cluster_analysis.to_excel(writer, sheet_name='Employee Clustering', startrow=4, startcol=0)
        
        # Cluster characteristics
        worksheet.write('A15', 'CLUSTER CHARACTERISTICS', header_format)
        worksheet.write('A16', 'Cluster')
        worksheet.write('B16', 'Size')
        worksheet.write('C16', 'Avg Performance')
        worksheet.write('D16', 'Top Departments')
        worksheet.write('E16', 'Performance Range')
        
        row = 17
        for cluster, characteristics in cluster_characteristics.items():
            worksheet.write(f'A{row}', cluster)
            worksheet.write(f'B{row}', characteristics['Size'])
            worksheet.write(f'C{row}', f"{characteristics['Avg Performance']:.1f}")
            worksheet.write(f'D{row}', str(characteristics['Top Departments']))
            worksheet.write(f'E{row}', characteristics['Performance Range'])
            row += 1
        
        # Employee cluster assignments
        worksheet.write(f'A{row+2}', 'EMPLOYEE CLUSTER ASSIGNMENTS', header_format)
        cluster_assignments = self.analyzer.employee_df[['employee_id', 'department', 'position', 'overall_performance_score', 'cluster']]
        cluster_assignments.to_excel(writer, sheet_name='Employee Clustering', startrow=row+3, startcol=0, index=False)
    
    def _generate_predictive_analysis(self, writer, header_format, title_format):
        """Generate predictive analysis sheet"""
        worksheet = writer.book.add_worksheet('Predictive Analysis')
        
        # Title
        worksheet.merge_range('A1:D1', 'PREDICTIVE PERFORMANCE ANALYSIS', title_format)
        
        # Run predictive model
        model_results = self.analyzer.performance_prediction_model()
        
        # Model performance metrics
        worksheet.write('A3', 'MODEL PERFORMANCE METRICS', header_format)
        worksheet.write('A4', 'Metric')
        worksheet.write('B4', 'Value')
        worksheet.write('C4', 'Description')
        
        row = 5
        worksheet.write(f'A{row}', 'R² Score')
        worksheet.write(f'B{row}', f"{model_results['r2_score']:.3f}")
        worksheet.write(f'C{row}', 'Coefficient of determination - higher is better')
        row += 1
        
        worksheet.write(f'A{row}', 'Mean Squared Error')
        worksheet.write(f'B{row}', f"{model_results['mse']:.3f}")
        worksheet.write(f'C{row}', 'Average squared prediction error - lower is better')
        row += 1
        
        worksheet.write(f'A{row}', 'Model Accuracy')
        worksheet.write(f'B{row}', f"{model_results['r2_score']*100:.1f}%")
        worksheet.write(f'C{row}', 'Percentage of variance explained by the model')
        
        # Feature importance
        worksheet.write(f'A{row+2}', 'FEATURE IMPORTANCE', header_format)
        worksheet.write(f'A{row+3}', 'Feature')
        worksheet.write(f'B{row+3}', 'Importance Score')
        worksheet.write(f'C{row+3}', 'Rank')
        
        row += 4
        for idx, (_, row_data) in enumerate(model_results['feature_importance'].iterrows(), 1):
            worksheet.write(f'A{row}', row_data['feature'].replace('_', ' ').title())
            worksheet.write(f'B{row}', f"{row_data['importance']:.4f}")
            worksheet.write(f'C{row}', idx)
            row += 1
    
    def _generate_trends_analysis(self, writer, header_format, title_format):
        """Generate trends analysis sheet"""
        worksheet = writer.book.add_worksheet('Trends Analysis')
        
        # Title
        worksheet.merge_range('A1:D1', 'PERFORMANCE TRENDS ANALYSIS', title_format)
        
        if self.analyzer.monthly_df is None:
            worksheet.write('A3', 'Monthly data not available for trend analysis.')
            return
        
        # Monthly performance trends
        monthly_trends = self.analyzer.monthly_df.groupby('month').agg({
            'overall_performance_score': ['mean', 'std'],
            'productivity_score': 'mean',
            'quality_score': 'mean',
            'teamwork_score': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_trends.columns = ['_'.join(col).strip() for col in monthly_trends.columns]
        monthly_trends.reset_index(inplace=True)
        
        worksheet.write('A3', 'MONTHLY PERFORMANCE TRENDS', header_format)
        monthly_trends.to_excel(writer, sheet_name='Trends Analysis', startrow=4, startcol=0, index=False)
        
        # Department trends
        worksheet.write(f'A{len(monthly_trends)+8}', 'DEPARTMENT PERFORMANCE TRENDS', header_format)
        dept_trends = self.analyzer.monthly_df.groupby(['month', 'department'])['overall_performance_score'].mean().unstack()
        dept_trends.to_excel(writer, sheet_name='Trends Analysis', startrow=len(monthly_trends)+9, startcol=0)
    
    def _generate_goals_analysis(self, writer, header_format, title_format):
        """Generate goals analysis sheet"""
        worksheet = writer.book.add_worksheet('Goals Analysis')
        
        # Title
        worksheet.merge_range('A1:E1', 'EMPLOYEE GOALS ANALYSIS', title_format)
        
        if self.analyzer.goals_df is None:
            worksheet.write('A3', 'Goals data not available.')
            return
        
        # Goals summary
        worksheet.write('A3', 'GOALS SUMMARY', header_format)
        worksheet.write('A4', 'Metric')
        worksheet.write('B4', 'Value')
        worksheet.write('C4', 'Description')
        
        row = 5
        total_goals = len(self.analyzer.goals_df)
        completed_goals = len(self.analyzer.goals_df[self.analyzer.goals_df['status'] == 'Completed'])
        avg_achievement = self.analyzer.goals_df['achievement_rate'].mean()
        goal_types = self.analyzer.goals_df['goal_type'].nunique()
        
        worksheet.write(f'A{row}', 'Total Goals')
        worksheet.write(f'B{row}', total_goals)
        worksheet.write(f'C{row}', 'Total number of goals set')
        row += 1
        
        worksheet.write(f'A{row}', 'Completed Goals')
        worksheet.write(f'B{row}', completed_goals)
        worksheet.write(f'C{row}', f"Goals with 100% achievement rate")
        row += 1
        
        worksheet.write(f'A{row}', 'Average Achievement Rate')
        worksheet.write(f'B{row}', f"{avg_achievement:.1f}%")
        worksheet.write(f'C{row}', 'Average achievement rate across all goals')
        row += 1
        
        worksheet.write(f'A{row}', 'Goal Types')
        worksheet.write(f'B{row}', goal_types)
        worksheet.write(f'C{row}', 'Number of different goal categories')
        
        # Goals by type
        worksheet.write(f'A{row+2}', 'GOALS BY TYPE', header_format)
        goal_type_counts = self.analyzer.goals_df['goal_type'].value_counts()
        worksheet.write(f'A{row+3}', 'Goal Type')
        worksheet.write(f'B{row+3}', 'Count')
        worksheet.write(f'C{row+3}', 'Percentage')
        
        row += 4
        for goal_type, count in goal_type_counts.items():
            percentage = (count / total_goals) * 100
            worksheet.write(f'A{row}', goal_type)
            worksheet.write(f'B{row}', count)
            worksheet.write(f'C{row}', f"{percentage:.1f}%")
            row += 1
        
        # Department goals analysis
        worksheet.write(f'A{row+2}', 'GOALS BY DEPARTMENT', header_format)
        dept_goals = self.analyzer.goals_df.groupby('department').agg({
            'achievement_rate': ['mean', 'count'],
            'status': lambda x: (x == 'Completed').sum()
        }).round(2)
        
        # Flatten column names
        dept_goals.columns = ['_'.join(col).strip() for col in dept_goals.columns]
        dept_goals.reset_index(inplace=True)
        dept_goals.columns = ['Department', 'Avg Achievement Rate', 'Total Goals', 'Completed Goals']
        
        dept_goals.to_excel(writer, sheet_name='Goals Analysis', startrow=row+3, startcol=0, index=False)
    
    def _generate_recommendations(self, writer, header_format, title_format):
        """Generate recommendations sheet"""
        worksheet = writer.book.add_worksheet('Recommendations')
        
        # Title
        worksheet.merge_range('A1:D1', 'PERFORMANCE IMPROVEMENT RECOMMENDATIONS', title_format)
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations_list()
        
        worksheet.write('A3', 'RECOMMENDATIONS', header_format)
        worksheet.write('A4', 'Category')
        worksheet.write('B4', 'Recommendation')
        worksheet.write('C4', 'Priority')
        worksheet.write('D4', 'Expected Impact')
        
        row = 5
        for category, recs in recommendations.items():
            for rec in recs:
                worksheet.write(f'A{row}', category)
                worksheet.write(f'B{row}', rec['recommendation'])
                worksheet.write(f'C{row}', rec['priority'])
                worksheet.write(f'D{row}', rec['impact'])
                row += 1
    
    def _generate_recommendations_list(self):
        """Generate recommendations based on analysis results"""
        recommendations = {
            'High Performers': [
                {
                    'recommendation': 'Implement mentorship programs for high performers to share knowledge',
                    'priority': 'High',
                    'impact': 'Knowledge transfer and team development'
                },
                {
                    'recommendation': 'Provide leadership development opportunities for A-grade employees',
                    'priority': 'High',
                    'impact': 'Succession planning and retention'
                }
            ],
            'Low Performers': [
                {
                    'recommendation': 'Develop targeted improvement plans for F-grade employees',
                    'priority': 'Critical',
                    'impact': 'Performance improvement and retention'
                },
                {
                    'recommendation': 'Provide additional training and support resources',
                    'priority': 'High',
                    'impact': 'Skill development and engagement'
                }
            ],
            'Department Optimization': [
                {
                    'recommendation': 'Analyze and address performance gaps between departments',
                    'priority': 'Medium',
                    'impact': 'Organizational alignment and efficiency'
                },
                {
                    'recommendation': 'Share best practices from top-performing departments',
                    'priority': 'Medium',
                    'impact': 'Cross-department learning'
                }
            ],
            'Training & Development': [
                {
                    'recommendation': 'Increase training hours for employees with low scores',
                    'priority': 'High',
                    'impact': 'Skill improvement and performance enhancement'
                },
                {
                    'recommendation': 'Implement skill-specific training programs',
                    'priority': 'Medium',
                    'impact': 'Targeted development'
                }
            ],
            'Goals Management': [
                {
                    'recommendation': 'Improve goal setting and tracking processes',
                    'priority': 'Medium',
                    'impact': 'Better goal achievement rates'
                },
                {
                    'recommendation': 'Provide regular feedback on goal progress',
                    'priority': 'High',
                    'impact': 'Increased motivation and accountability'
                }
            ]
        }
        
        return recommendations

def main():
    """Main function to generate report"""
    # Initialize analyzer
    analyzer = EmployeePerformanceAnalyzer()
    
    try:
        analyzer.load_data('employee_data.csv', 'monthly_performance.csv', 'employee_goals.csv')
        
        # Generate report
        report_generator = PerformanceReportGenerator(analyzer)
        filename = report_generator.generate_excel_report()
        
        print(f"✅ Comprehensive report generated: {filename}")
        print("📊 Report includes:")
        print("   - Executive Summary")
        print("   - Department Analysis")
        print("   - Performance Metrics")
        print("   - Employee Clustering")
        print("   - Predictive Analysis")
        print("   - Trends Analysis")
        print("   - Goals Analysis")
        print("   - Recommendations")
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run data_generator.py first to generate sample data.")

if __name__ == "__main__":
    main() 