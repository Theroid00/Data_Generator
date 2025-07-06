"""
Dataset Manager Module
Handles collection, management, and analysis of training datasets
Creates preset templates from real data patterns
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile
import requests
from datetime import datetime


class DatasetManager:
    """Manages dataset collection and template creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.datasets_dir = Path('datasets')
        self.templates_dir = Path('templates')
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        self.datasets_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        
    def download_sample_datasets(self) -> List[Path]:
        """Download various sample datasets for training"""
        print("🔹 Downloading sample datasets...")
        downloaded = []
        
        # Create diverse sample datasets
        datasets_to_create = [
            ('employees', self._create_employee_dataset),
            ('customers', self._create_customer_dataset),
            ('products', self._create_product_dataset),
            ('transactions', self._create_transaction_dataset),
            ('healthcare', self._create_healthcare_dataset),
            ('education', self._create_education_dataset),
            ('financial', self._create_financial_dataset),
            ('ecommerce', self._create_ecommerce_dataset)
        ]
        
        for name, creator_func in datasets_to_create:
            try:
                path = creator_func()
                downloaded.append(path)
                print(f"✅ Created {name} dataset: {path}")
            except Exception as e:
                print(f"❌ Failed to create {name} dataset: {e}")
                
        return downloaded
    
    def _create_employee_dataset(self) -> Path:
        """Create comprehensive employee dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 2000
        departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations', 'Legal', 'Design']
        job_levels = ['Junior', 'Senior', 'Lead', 'Manager', 'Director', 'VP']
        
        data = {
            'employee_id': range(1000, 1000 + n_rows),
            'first_name': [fake.first_name() for _ in range(n_rows)],
            'last_name': [fake.last_name() for _ in range(n_rows)],
            'email': [],
            'department': np.random.choice(departments, n_rows),
            'job_level': np.random.choice(job_levels, n_rows),
            'salary': [],
            'hire_date': [fake.date_between(start_date='-10y', end_date='today') for _ in range(n_rows)],
            'age': np.random.normal(35, 10, n_rows).astype(int).clip(22, 67),
            'performance_rating': np.random.choice(['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'], n_rows, p=[0.2, 0.4, 0.3, 0.1]),
            'remote_work': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
            'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'Australia'], n_rows),
            'years_experience': []
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Email from name
            email = f"{data['first_name'][i].lower()}.{data['last_name'][i].lower()}@company.com"
            data['email'].append(email)
            
            # Salary based on level and department
            base_salary = {
                'Junior': 50000, 'Senior': 75000, 'Lead': 95000,
                'Manager': 110000, 'Director': 150000, 'VP': 200000
            }
            dept_multiplier = {
                'Engineering': 1.2, 'Finance': 1.1, 'Sales': 1.0,
                'Marketing': 0.95, 'HR': 0.9, 'Operations': 0.9,
                'Legal': 1.15, 'Design': 1.05
            }
            
            salary = base_salary[data['job_level'][i]] * dept_multiplier[data['department'][i]]
            salary += np.random.normal(0, 10000)  # Add variation
            data['salary'].append(max(35000, int(salary)))
            
            # Years experience based on age
            min_exp = max(0, data['age'][i] - 22)
            max_exp = min(min_exp + 15, data['age'][i] - 18)
            data['years_experience'].append(np.random.randint(min_exp, max_exp + 1))
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'employees_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_customer_dataset(self) -> Path:
        """Create customer dataset with purchase behavior"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 1500
        
        data = {
            'customer_id': [f"CUST_{i:06d}" for i in range(1, n_rows + 1)],
            'full_name': [fake.name() for _ in range(n_rows)],
            'email': [fake.email() for _ in range(n_rows)],
            'phone': [fake.phone_number() for _ in range(n_rows)],
            'age': np.random.normal(40, 15, n_rows).astype(int).clip(18, 80),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_rows, p=[0.48, 0.48, 0.04]),
            'registration_date': [fake.date_between(start_date='-5y', end_date='today') for _ in range(n_rows)],
            'last_purchase_date': [],
            'total_spent': [],
            'purchase_frequency': np.random.choice(['Weekly', 'Monthly', 'Quarterly', 'Rarely'], n_rows, p=[0.15, 0.35, 0.35, 0.15]),
            'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_rows),
            'loyalty_tier': [],
            'country': np.random.choice(['USA', 'Canada', 'UK', 'Australia', 'Germany'], n_rows),
            'city': [],
            'subscription_status': np.random.choice(['Active', 'Inactive', 'Churned'], n_rows, p=[0.6, 0.25, 0.15])
        }
        
        # Generate derived fields
        country_cities = {
            'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa'],
            'UK': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow'],
            'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
            'Germany': ['Berlin', 'Munich', 'Hamburg', 'Cologne', 'Frankfurt']
        }
        
        for i in range(n_rows):
            # City based on country
            data['city'].append(np.random.choice(country_cities[data['country'][i]]))
            
            # Total spent based on age and frequency
            base_spend = np.random.uniform(100, 5000)
            freq_multiplier = {'Weekly': 4, 'Monthly': 2, 'Quarterly': 1, 'Rarely': 0.3}
            total_spent = base_spend * freq_multiplier[data['purchase_frequency'][i]]
            data['total_spent'].append(round(total_spent, 2))
            
            # Loyalty tier based on spending
            if total_spent > 3000:
                tier = 'Platinum'
            elif total_spent > 1500:
                tier = 'Gold'
            elif total_spent > 500:
                tier = 'Silver'
            else:
                tier = 'Bronze'
            data['loyalty_tier'].append(tier)
            
            # Last purchase relative to registration
            reg_date = pd.to_datetime(data['registration_date'][i])
            days_since_reg = (datetime.now() - reg_date).days
            last_purchase_days = np.random.randint(0, min(days_since_reg, 365))
            last_purchase = datetime.now() - pd.Timedelta(days=last_purchase_days)
            data['last_purchase_date'].append(last_purchase.date())
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'customers_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_product_dataset(self) -> Path:
        """Create product catalog dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 1000
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Automotive']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Generic']
        
        data = {
            'product_id': [f"PROD_{i:05d}" for i in range(1, n_rows + 1)],
            'product_name': [fake.catch_phrase() for _ in range(n_rows)],
            'category': np.random.choice(categories, n_rows),
            'brand': np.random.choice(brands, n_rows),
            'price': [],
            'cost': [],
            'stock_quantity': np.random.randint(0, 1000, n_rows),
            'weight_kg': [],
            'rating': np.random.uniform(1, 5, n_rows).round(1),
            'review_count': np.random.randint(0, 5000, n_rows),
            'launch_date': [fake.date_between(start_date='-3y', end_date='today') for _ in range(n_rows)],
            'is_bestseller': [],
            'supplier_country': np.random.choice(['China', 'USA', 'Germany', 'Japan', 'South Korea'], n_rows)
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Price based on category
            category_price_ranges = {
                'Electronics': (50, 2000),
                'Clothing': (10, 300),
                'Home & Garden': (20, 500),
                'Books': (5, 50),
                'Sports': (15, 800),
                'Beauty': (8, 200),
                'Automotive': (25, 1500)
            }
            
            min_price, max_price = category_price_ranges[data['category'][i]]
            price = round(np.random.uniform(min_price, max_price), 2)
            data['price'].append(price)
            
            # Cost is 60-80% of price
            cost = round(price * np.random.uniform(0.6, 0.8), 2)
            data['cost'].append(cost)
            
            # Weight based on category
            if data['category'][i] == 'Electronics':
                weight = np.random.uniform(0.1, 5.0)
            elif data['category'][i] == 'Clothing':
                weight = np.random.uniform(0.1, 2.0)
            else:
                weight = np.random.uniform(0.1, 10.0)
            data['weight_kg'].append(round(weight, 2))
            
            # Bestseller based on rating and reviews
            is_bestseller = data['rating'][i] > 4.0 and data['review_count'][i] > 100
            data['is_bestseller'].append(is_bestseller)
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'products_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_transaction_dataset(self) -> Path:
        """Create transaction/sales dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 3000
        
        data = {
            'transaction_id': [f"TXN_{i:08d}" for i in range(1, n_rows + 1)],
            'customer_id': [f"CUST_{np.random.randint(1, 1501):06d}" for _ in range(n_rows)],
            'product_id': [f"PROD_{np.random.randint(1, 1001):05d}" for _ in range(n_rows)],
            'transaction_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_rows)],
            'quantity': np.random.randint(1, 10, n_rows),
            'unit_price': [],
            'total_amount': [],
            'discount_applied': np.random.uniform(0, 0.3, n_rows).round(2),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'shipping_cost': [],
            'tax_amount': [],
            'order_status': np.random.choice(['Completed', 'Pending', 'Shipped', 'Cancelled'], n_rows, p=[0.7, 0.1, 0.15, 0.05]),
            'return_flag': np.random.choice([True, False], n_rows, p=[0.05, 0.95])
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Unit price varies by product
            unit_price = round(np.random.uniform(10, 500), 2)
            data['unit_price'].append(unit_price)
            
            # Total before discount
            subtotal = unit_price * data['quantity'][i]
            discount_amount = subtotal * data['discount_applied'][i]
            
            # Shipping cost
            shipping = round(np.random.uniform(0, 25), 2) if subtotal < 100 else 0
            data['shipping_cost'].append(shipping)
            
            # Tax (8% of subtotal after discount)
            tax = round((subtotal - discount_amount) * 0.08, 2)
            data['tax_amount'].append(tax)
            
            # Final total
            total = subtotal - discount_amount + shipping + tax
            data['total_amount'].append(round(total, 2))
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'transactions_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_healthcare_dataset(self) -> Path:
        """Create healthcare/medical dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 1200
        
        data = {
            'patient_id': [f"PAT_{i:06d}" for i in range(1, n_rows + 1)],
            'age': np.random.normal(45, 20, n_rows).astype(int).clip(1, 95),
            'gender': np.random.choice(['Male', 'Female'], n_rows),
            'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_rows, p=[0.34, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.07]),
            'height_cm': [],
            'weight_kg': [],
            'bmi': [],
            'blood_pressure_systolic': [],
            'blood_pressure_diastolic': [],
            'cholesterol_level': [],
            'glucose_level': [],
            'diagnosis': np.random.choice(['Healthy', 'Hypertension', 'Diabetes', 'Heart Disease', 'Obesity'], n_rows, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
            'admission_date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_rows)],
            'discharge_date': [],
            'insurance_provider': np.random.choice(['BlueCross', 'Aetna', 'Kaiser', 'UnitedHealth', 'Medicare'], n_rows),
            'treatment_cost': []
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Height based on gender and age
            if data['gender'][i] == 'Male':
                height = np.random.normal(175, 8)  # cm
            else:
                height = np.random.normal(162, 7)  # cm
            data['height_cm'].append(round(height, 1))
            
            # Weight correlated with height and age
            base_weight = (height / 100) ** 2 * 22  # Normal BMI baseline
            weight_variation = np.random.normal(0, 10)
            weight = max(30, base_weight + weight_variation)
            data['weight_kg'].append(round(weight, 1))
            
            # BMI calculation
            bmi = weight / ((height / 100) ** 2)
            data['bmi'].append(round(bmi, 1))
            
            # Blood pressure (age-related)
            age = data['age'][i]
            systolic = np.random.normal(120 + age * 0.5, 15)
            diastolic = np.random.normal(80 + age * 0.2, 10)
            data['blood_pressure_systolic'].append(int(max(90, min(200, systolic))))
            data['blood_pressure_diastolic'].append(int(max(60, min(120, diastolic))))
            
            # Cholesterol (age and weight related)
            cholesterol = np.random.normal(180 + age * 0.8 + (bmi - 25) * 2, 30)
            data['cholesterol_level'].append(int(max(120, min(350, cholesterol))))
            
            # Glucose level
            if data['diagnosis'][i] == 'Diabetes':
                glucose = np.random.normal(150, 25)
            else:
                glucose = np.random.normal(95, 15)
            data['glucose_level'].append(int(max(70, min(300, glucose))))
            
            # Discharge date (1-10 days after admission)
            admission = pd.to_datetime(data['admission_date'][i])
            stay_days = np.random.randint(1, 11)
            discharge = admission + pd.Timedelta(days=stay_days)
            data['discharge_date'].append(discharge.date())
            
            # Treatment cost based on diagnosis and stay
            base_costs = {
                'Healthy': 500,
                'Hypertension': 1200,
                'Diabetes': 1800,
                'Heart Disease': 3500,
                'Obesity': 1000
            }
            cost = base_costs[data['diagnosis'][i]] * stay_days + np.random.uniform(-200, 500)
            data['treatment_cost'].append(round(max(200, cost), 2))
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'healthcare_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_education_dataset(self) -> Path:
        """Create education/student dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 800
        
        data = {
            'student_id': [f"STU_{i:05d}" for i in range(1, n_rows + 1)],
            'first_name': [fake.first_name() for _ in range(n_rows)],
            'last_name': [fake.last_name() for _ in range(n_rows)],
            'age': np.random.randint(16, 25, n_rows),
            'grade_level': [],
            'gpa': [],
            'major': np.random.choice(['Computer Science', 'Business', 'Engineering', 'Biology', 'Psychology', 'Mathematics', 'English', 'History'], n_rows),
            'enrollment_date': [fake.date_between(start_date='-4y', end_date='today') for _ in range(n_rows)],
            'expected_graduation': [],
            'tuition_fee': [],
            'scholarship_amount': [],
            'attendance_rate': np.random.uniform(0.7, 1.0, n_rows).round(2),
            'extracurricular_activities': np.random.randint(0, 8, n_rows),
            'dormitory_resident': np.random.choice([True, False], n_rows, p=[0.6, 0.4]),
            'home_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_rows)
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Grade level based on age
            age = data['age'][i]
            if age <= 18:
                grade = 'Freshman'
            elif age <= 20:
                grade = 'Sophomore'
            elif age <= 22:
                grade = 'Junior'
            else:
                grade = 'Senior'
            data['grade_level'].append(grade)
            
            # GPA with some correlation to attendance
            base_gpa = 2.5 + data['attendance_rate'][i] * 1.5
            gpa_variation = np.random.normal(0, 0.3)
            gpa = max(0.0, min(4.0, base_gpa + gpa_variation))
            data['gpa'].append(round(gpa, 2))
            
            # Expected graduation based on enrollment and current grade
            enrollment = pd.to_datetime(data['enrollment_date'][i])
            years_to_graduate = {'Freshman': 4, 'Sophomore': 3, 'Junior': 2, 'Senior': 1}
            graduation = enrollment + pd.DateOffset(years=years_to_graduate[grade])
            data['expected_graduation'].append(graduation.date())
            
            # Tuition based on major and residency
            base_tuition = {
                'Computer Science': 45000, 'Engineering': 43000, 'Business': 40000,
                'Biology': 38000, 'Mathematics': 35000, 'Psychology': 33000,
                'English': 30000, 'History': 28000
            }
            tuition = base_tuition[data['major'][i]]
            if data['home_state'][i] in ['CA', 'NY', 'TX']:  # Higher cost states
                tuition *= 1.2
            data['tuition_fee'].append(tuition)
            
            # Scholarship based on GPA
            if gpa >= 3.8:
                scholarship = np.random.uniform(5000, 15000)
            elif gpa >= 3.5:
                scholarship = np.random.uniform(2000, 8000)
            elif gpa >= 3.0:
                scholarship = np.random.uniform(0, 3000)
            else:
                scholarship = 0
            data['scholarship_amount'].append(round(scholarship))
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'education_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_financial_dataset(self) -> Path:
        """Create financial/banking dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 1800
        
        data = {
            'account_id': [f"ACC_{i:08d}" for i in range(1, n_rows + 1)],
            'customer_name': [fake.name() for _ in range(n_rows)],
            'account_type': np.random.choice(['Checking', 'Savings', 'Credit', 'Investment'], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'balance': [],
            'credit_score': np.random.normal(700, 80, n_rows).astype(int).clip(300, 850),
            'annual_income': [],
            'account_opening_date': [fake.date_between(start_date='-10y', end_date='today') for _ in range(n_rows)],
            'last_transaction_date': [],
            'transaction_count_monthly': np.random.poisson(15, n_rows),
            'overdraft_count': np.random.poisson(1, n_rows),
            'loan_amount': [],
            'loan_interest_rate': [],
            'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], n_rows, p=[0.65, 0.15, 0.05, 0.15]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_rows, p=[0.35, 0.45, 0.15, 0.05])
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Annual income based on employment status and credit score
            if data['employment_status'][i] == 'Employed':
                base_income = 30000 + (data['credit_score'][i] - 300) * 200
            elif data['employment_status'][i] == 'Self-employed':
                base_income = 25000 + (data['credit_score'][i] - 300) * 150
            elif data['employment_status'][i] == 'Retired':
                base_income = 20000 + (data['credit_score'][i] - 300) * 50
            else:  # Unemployed
                base_income = 5000
            
            income_variation = np.random.normal(0, 10000)
            income = max(0, base_income + income_variation)
            data['annual_income'].append(int(income))
            
            # Balance based on account type and income
            if data['account_type'][i] == 'Checking':
                balance = np.random.uniform(100, income * 0.1)
            elif data['account_type'][i] == 'Savings':
                balance = np.random.uniform(500, income * 0.3)
            elif data['account_type'][i] == 'Investment':
                balance = np.random.uniform(1000, income * 0.5)
            else:  # Credit
                balance = -np.random.uniform(0, 5000)  # Negative balance
            
            data['balance'].append(round(balance, 2))
            
            # Last transaction (within last 30 days)
            last_transaction = fake.date_between(start_date='-30d', end_date='today')
            data['last_transaction_date'].append(last_transaction)
            
            # Loan amount based on income and credit score
            if np.random.random() < 0.3:  # 30% have loans
                max_loan = income * 3 + (data['credit_score'][i] - 300) * 100
                loan = np.random.uniform(1000, max_loan)
                interest_rate = max(2.5, 15 - (data['credit_score'][i] - 300) / 100)
            else:
                loan = 0
                interest_rate = 0
            
            data['loan_amount'].append(round(loan, 2))
            data['loan_interest_rate'].append(round(interest_rate, 2))
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'financial_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def _create_ecommerce_dataset(self) -> Path:
        """Create e-commerce dataset"""
        from faker import Faker
        fake = Faker()
        
        n_rows = 2500
        
        data = {
            'order_id': [f"ORD_{i:07d}" for i in range(1, n_rows + 1)],
            'customer_email': [fake.email() for _ in range(n_rows)],
            'order_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_rows)],
            'product_category': np.random.choice(['Fashion', 'Electronics', 'Home', 'Beauty', 'Sports', 'Books'], n_rows),
            'product_subcategory': [],
            'order_value': [],
            'shipping_cost': [],
            'discount_percentage': np.random.uniform(0, 0.5, n_rows).round(2),
            'delivery_days': np.random.randint(1, 15, n_rows),
            'customer_satisfaction': np.random.randint(1, 6, n_rows),  # 1-5 scale
            'return_requested': np.random.choice([True, False], n_rows, p=[0.08, 0.92]),
            'marketing_channel': np.random.choice(['Organic', 'Paid Search', 'Social Media', 'Email', 'Referral'], n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_rows, p=[0.6, 0.3, 0.1]),
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card', 'Digital Wallet'], n_rows, p=[0.45, 0.25, 0.2, 0.1])
        }
        
        # Define subcategories
        subcategories = {
            'Fashion': ['Clothing', 'Shoes', 'Accessories', 'Jewelry'],
            'Electronics': ['Smartphones', 'Laptops', 'Audio', 'Gaming'],
            'Home': ['Furniture', 'Decor', 'Kitchen', 'Garden'],
            'Beauty': ['Skincare', 'Makeup', 'Fragrance', 'Hair Care'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports'],
            'Books': ['Fiction', 'Non-fiction', 'Educational', 'Children']
        }
        
        # Generate derived fields
        for i in range(n_rows):
            # Subcategory based on category
            category = data['product_category'][i]
            subcategory = np.random.choice(subcategories[category])
            data['product_subcategory'].append(subcategory)
            
            # Order value based on category
            category_price_ranges = {
                'Fashion': (25, 300),
                'Electronics': (50, 1500),
                'Home': (30, 800),
                'Beauty': (15, 150),
                'Sports': (20, 400),
                'Books': (8, 60)
            }
            
            min_price, max_price = category_price_ranges[category]
            order_value = np.random.uniform(min_price, max_price)
            data['order_value'].append(round(order_value, 2))
            
            # Shipping cost based on order value
            if order_value > 100:
                shipping = 0  # Free shipping
            elif order_value > 50:
                shipping = round(np.random.uniform(3, 8), 2)
            else:
                shipping = round(np.random.uniform(5, 15), 2)
            
            data['shipping_cost'].append(shipping)
        
        df = pd.DataFrame(data)
        path = self.datasets_dir / 'ecommerce_comprehensive.csv'
        df.to_csv(path, index=False)
        return path
    
    def analyze_dataset(self, csv_path: Path) -> Dict:
        """Analyze a dataset to extract patterns and create template"""
        try:
            df = pd.read_csv(csv_path)
            
            analysis = {
                'dataset_name': csv_path.stem,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'num_rows': len(df),  # Keep both for backward compatibility
                'num_columns': len(df.columns),
                'columns': {},
                'relationships': [],
                'quality_score': 0,
                'created_at': datetime.now().isoformat()
            }
            
            # Analyze each column
            for column in df.columns:
                col_analysis = self._analyze_column(df[column], column)
                analysis['columns'][column] = col_analysis
            
            # Detect relationships
            analysis['relationships'] = self._detect_relationships(df)
            
            # Calculate quality score
            analysis['quality_score'] = self._calculate_quality_score(df)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Dataset analysis failed: {e}")
            return {}
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict:
        """Analyze individual column"""
        analysis = {
            'name': column_name,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100,
            'unique_count': series.nunique(),
            'data_type': 'unknown'
        }
        
        # Determine semantic data type
        if pd.api.types.is_numeric_dtype(series):
            analysis['data_type'] = 'integer' if series.dtype == 'int64' else 'float'
            analysis['min'] = series.min()
            analysis['max'] = series.max()
            analysis['mean'] = series.mean()
            analysis['std'] = series.std()
            
        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis['data_type'] = 'date'
            analysis['min'] = series.min()
            analysis['max'] = series.max()
            
        else:
            # Categorical or text analysis
            if analysis['unique_count'] / len(series) < 0.1:
                analysis['data_type'] = 'categorical'
                analysis['categories'] = series.value_counts().head(20).to_dict()
            else:
                # Check for specific patterns
                if 'email' in column_name.lower():
                    analysis['data_type'] = 'email'
                elif 'phone' in column_name.lower():
                    analysis['data_type'] = 'phone'
                elif 'name' in column_name.lower():
                    analysis['data_type'] = 'name'
                elif 'address' in column_name.lower():
                    analysis['data_type'] = 'address'
                else:
                    analysis['data_type'] = 'text'
                
                analysis['sample_values'] = series.dropna().head(5).tolist()
        
        return analysis
    
    def _detect_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Detect relationships between columns"""
        relationships = []
        
        # Common relationship patterns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if len(date_columns) >= 2:
            relationships.append({
                'type': 'date_sequence',
                'columns': date_columns,
                'description': 'Date columns should maintain chronological order'
            })
        
        # Geographic relationships
        geo_columns = {
            'country': [col for col in df.columns if 'country' in col.lower()],
            'city': [col for col in df.columns if 'city' in col.lower()],
            'state': [col for col in df.columns if 'state' in col.lower()]
        }
        
        if geo_columns['country'] and geo_columns['city']:
            relationships.append({
                'type': 'geographic',
                'columns': geo_columns['country'] + geo_columns['city'],
                'description': 'Country and city should be geographically consistent'
            })
        
        # Financial relationships
        financial_columns = [col for col in df.columns if any(term in col.lower() for term in ['salary', 'income', 'price', 'cost', 'amount'])]
        if len(financial_columns) >= 2:
            relationships.append({
                'type': 'financial',
                'columns': financial_columns,
                'description': 'Financial amounts should be realistic and correlated'
            })
        
        return relationships
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate dataset quality score (0-100)"""
        score = 100
        
        # Penalize for missing data
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        score -= missing_percentage * 2
        
        # Penalize for very low or very high cardinality
        for column in df.columns:
            unique_ratio = df[column].nunique() / len(df)
            if unique_ratio < 0.01 or unique_ratio > 0.95:  # Too uniform or too diverse
                score -= 5
        
        # Bonus for reasonable size
        if 100 <= len(df) <= 10000:
            score += 5
        
        return max(0, min(100, score))
    
    def create_preset_template(self, dataset_analysis: Dict, template_name: str) -> Path:
        """Create a preset template from dataset analysis"""
        template = {
            'name': template_name,
            'description': f"Template based on {dataset_analysis['dataset_name']} dataset",
            'source_dataset': dataset_analysis['dataset_name'],
            'created_at': datetime.now().isoformat(),
            'quality_score': dataset_analysis['quality_score'],
            'columns': [],
            'relationships': dataset_analysis['relationships']
        }
        
        # Convert analysis to template format
        for col_name, col_info in dataset_analysis['columns'].items():
            template_col = {
                'name': col_name,
                'data_type': col_info['data_type'],
                'required': col_info['null_percentage'] < 10,  # Required if < 10% null
                'description': f"Auto-generated from {dataset_analysis['dataset_name']}"
            }
            
            # Add type-specific parameters
            if col_info['data_type'] in ['integer', 'float']:
                template_col['min_value'] = float(col_info.get('min', 0))
                template_col['max_value'] = float(col_info.get('max', 100))
                template_col['distribution'] = 'normal'
                
            elif col_info['data_type'] == 'categorical':
                template_col['categories'] = list(col_info.get('categories', {}).keys())
                
            elif col_info['data_type'] == 'date':
                template_col['min_value'] = str(col_info.get('min', '2020-01-01'))[:10]  # Keep only date part
                template_col['max_value'] = str(col_info.get('max', '2024-12-31'))[:10]  # Keep only date part
            
            template['columns'].append(template_col)
        
        # Save template
        template_path = self.templates_dir / f"{template_name}.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2, default=str)
        
        print(f"✅ Template created: {template_path}")
        return template_path
    
    def list_available_datasets(self) -> List[Path]:
        """List all available datasets"""
        return list(self.datasets_dir.glob('*.csv'))
    
    def list_available_templates(self) -> List[Path]:
        """List all available templates"""
        return list(self.templates_dir.glob('*.json'))


def main():
    """Demo the dataset manager"""
    manager = DatasetManager()
    
    print("🔹 Dataset Manager Demo")
    print("=" * 40)
    
    # Download datasets
    datasets = manager.download_sample_datasets()
    
    print(f"\n📊 Created {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"   {dataset}")
    
    # Analyze datasets and create templates
    print("\n🔍 Analyzing datasets and creating templates...")
    for dataset in datasets[:3]:  # Analyze first 3
        try:
            analysis = manager.analyze_dataset(dataset)
            if analysis:
                template_name = f"preset_{analysis['dataset_name']}"
                template_path = manager.create_preset_template(analysis, template_name)
                print(f"   📋 Template: {template_path}")
        except Exception as e:
            print(f"   ❌ Failed to analyze {dataset}: {e}")
    
    print("\n✅ Dataset Manager demo completed!")
    print(f"📁 Datasets: {len(manager.list_available_datasets())}")
    print(f"📋 Templates: {len(manager.list_available_templates())}")


if __name__ == "__main__":
    main()
