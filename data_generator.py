import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EmployeeDataGenerator:
    def __init__(self):
        self.departments = [
            'Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 
            'Operations', 'Customer Support', 'Product Management'
        ]
        
        self.positions = {
            'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Architect'],
            'Sales': ['Sales Representative', 'Account Manager', 'Sales Manager', 'VP Sales'],
            'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Brand Manager', 'CMO'],
            'HR': ['HR Specialist', 'HR Manager', 'Recruiter', 'HR Director'],
            'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'CFO'],
            'Operations': ['Operations Specialist', 'Operations Manager', 'Process Analyst'],
            'Customer Support': ['Support Specialist', 'Team Lead', 'Support Manager'],
            'Product Management': ['Product Manager', 'Senior PM', 'Product Director']
        }
        
        self.performance_metrics = [
            'productivity_score', 'quality_score', 'attendance_rate', 
            'teamwork_score', 'initiative_score', 'communication_score',
            'problem_solving_score', 'adaptability_score', 'leadership_score'
        ]
    
    def generate_employee_data(self, num_employees=100, start_date='2023-01-01', end_date='2024-01-01'):
        """Generate comprehensive employee performance data"""
        
        # Generate employee IDs and basic info
        employee_ids = [f"EMP{i:04d}" for i in range(1, num_employees + 1)]
        
        # Generate departments and positions
        departments = []
        positions = []
        for _ in range(num_employees):
            dept = random.choice(self.departments)
            departments.append(dept)
            positions.append(random.choice(self.positions[dept]))
        
        # Generate hire dates (within last 5 years)
        hire_dates = []
        for _ in range(num_employees):
            days_back = random.randint(0, 1825)  # 5 years
            hire_date = datetime.now() - timedelta(days=days_back)
            hire_dates.append(hire_date.strftime('%Y-%m-%d'))
        
        # Generate performance data
        performance_data = {}
        for metric in self.performance_metrics:
            # Generate scores with realistic distributions
            if 'productivity' in metric:
                scores = np.random.normal(75, 15, num_employees)
            elif 'quality' in metric:
                scores = np.random.normal(80, 12, num_employees)
            elif 'attendance' in metric:
                scores = np.random.normal(92, 8, num_employees)
            elif 'teamwork' in metric:
                scores = np.random.normal(78, 14, num_employees)
            elif 'initiative' in metric:
                scores = np.random.normal(70, 18, num_employees)
            elif 'communication' in metric:
                scores = np.random.normal(75, 16, num_employees)
            elif 'problem_solving' in metric:
                scores = np.random.normal(72, 17, num_employees)
            elif 'adaptability' in metric:
                scores = np.random.normal(76, 15, num_employees)
            elif 'leadership' in metric:
                scores = np.random.normal(65, 20, num_employees)
            
            # Clip scores to 0-100 range
            scores = np.clip(scores, 0, 100)
            performance_data[metric] = scores
        
        # Generate additional metrics
        performance_data['projects_completed'] = np.random.poisson(8, num_employees)
        performance_data['training_hours'] = np.random.normal(40, 15, num_employees)
        performance_data['overtime_hours'] = np.random.exponential(10, num_employees)
        performance_data['client_satisfaction'] = np.random.normal(85, 12, num_employees)
        
        # Create DataFrame
        df = pd.DataFrame({
            'employee_id': employee_ids,
            'department': departments,
            'position': positions,
            'hire_date': hire_dates,
            **performance_data
        })
        
        # Add calculated fields
        df['tenure_months'] = ((datetime.now() - pd.to_datetime(df['hire_date'])).dt.days / 30).astype(int)
        df['overall_performance_score'] = df[[col for col in df.columns if 'score' in col]].mean(axis=1)
        df['performance_grade'] = pd.cut(df['overall_performance_score'], 
                                       bins=[0, 60, 70, 80, 90, 100], 
                                       labels=['F', 'D', 'C', 'B', 'A'])
        
        return df
    
    def generate_monthly_data(self, employee_df, start_date='2023-01-01', end_date='2024-01-01'):
        """Generate monthly performance tracking data"""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        monthly_data = []
        
        for _, employee in employee_df.iterrows():
            current_date = start_dt
            while current_date <= end_dt:
                # Add some variation to monthly performance
                variation = np.random.normal(0, 5)
                
                monthly_record = {
                    'employee_id': employee['employee_id'],
                    'department': employee['department'],
                    'position': employee['position'],
                    'month': current_date.strftime('%Y-%m'),
                    'productivity_score': max(0, min(100, employee['productivity_score'] + variation)),
                    'quality_score': max(0, min(100, employee['quality_score'] + variation)),
                    'attendance_rate': max(0, min(100, employee['attendance_rate'] + variation)),
                    'teamwork_score': max(0, min(100, employee['teamwork_score'] + variation)),
                    'initiative_score': max(0, min(100, employee['initiative_score'] + variation)),
                    'communication_score': max(0, min(100, employee['communication_score'] + variation)),
                    'problem_solving_score': max(0, min(100, employee['problem_solving_score'] + variation)),
                    'adaptability_score': max(0, min(100, employee['adaptability_score'] + variation)),
                    'leadership_score': max(0, min(100, employee['leadership_score'] + variation)),
                    'projects_completed': max(0, int(employee['projects_completed'] / 12 + np.random.poisson(0.5))),
                    'training_hours': max(0, employee['training_hours'] / 12 + np.random.normal(0, 2)),
                    'overtime_hours': max(0, employee['overtime_hours'] / 12 + np.random.exponential(1)),
                    'client_satisfaction': max(0, min(100, employee['client_satisfaction'] + variation))
                }
                
                monthly_data.append(monthly_record)
                current_date += timedelta(days=32)  # Approximate month
                current_date = current_date.replace(day=1)  # Reset to first of month
        
        monthly_df = pd.DataFrame(monthly_data)
        monthly_df['overall_performance_score'] = monthly_df[[col for col in monthly_df.columns if 'score' in col]].mean(axis=1)
        
        return monthly_df
    
    def generate_goals_data(self, employee_df):
        """Generate employee goals and achievements data"""
        
        goals_data = []
        
        for _, employee in employee_df.iterrows():
            # Generate 3-5 goals per employee
            num_goals = random.randint(3, 5)
            
            for i in range(num_goals):
                goal_types = ['Performance Improvement', 'Skill Development', 'Project Completion', 
                            'Leadership Development', 'Client Satisfaction', 'Process Optimization']
                
                goal_type = random.choice(goal_types)
                target_value = random.randint(70, 95)
                current_value = random.randint(50, target_value + 10)
                achievement_rate = min(100, (current_value / target_value) * 100)
                
                goal_record = {
                    'employee_id': employee['employee_id'],
                    'department': employee['department'],
                    'goal_id': f"GOAL_{employee['employee_id']}_{i+1}",
                    'goal_type': goal_type,
                    'goal_description': f"Improve {goal_type.lower()} by achieving {target_value}% target",
                    'target_value': target_value,
                    'current_value': current_value,
                    'achievement_rate': achievement_rate,
                    'deadline': (datetime.now() + timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
                    'status': 'In Progress' if achievement_rate < 100 else 'Completed'
                }
                
                goals_data.append(goal_record)
        
        return pd.DataFrame(goals_data)

if __name__ == "__main__":
    # Generate sample data
    generator = EmployeeDataGenerator()
    
    # Generate employee data
    employee_df = generator.generate_employee_data(num_employees=150)
    
    # Generate monthly tracking data
    monthly_df = generator.generate_monthly_data(employee_df)
    
    # Generate goals data
    goals_df = generator.generate_goals_data(employee_df)
    
    # Save to files
    employee_df.to_csv('employee_data.csv', index=False)
    monthly_df.to_csv('monthly_performance.csv', index=False)
    goals_df.to_csv('employee_goals.csv', index=False)
    
    print("Data generation completed!")
    print(f"Generated {len(employee_df)} employee records")
    print(f"Generated {len(monthly_df)} monthly performance records")
    print(f"Generated {len(goals_df)} goal records") 