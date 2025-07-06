"""
Logic Rules Module
Handles data validation and relationship enforcement for synthetic data
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class DataValidator:
    """Validates data quality and logical consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate overall dataframe quality"""
        try:
            # Check for empty dataframe
            if df.empty:
                self.logger.error("Dataframe is empty")
                return False
            
            # Check for all null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                self.logger.warning(f"Columns with all null values: {null_columns}")
            
            # Check data types consistency
            for column in df.columns:
                if self._validate_column_consistency(df[column], column):
                    continue
                else:
                    self.logger.warning(f"Inconsistent data in column: {column}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False
    
    def _validate_column_consistency(self, series: pd.Series, column_name: str) -> bool:
        """Validate individual column consistency"""
        try:
            # Check for mixed types (excluding NaN)
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return True
            
            # Email validation
            if 'email' in column_name.lower():
                return self._validate_emails(non_null_series)
            
            # Phone validation
            if 'phone' in column_name.lower():
                return self._validate_phones(non_null_series)
            
            # Date validation
            if 'date' in column_name.lower() or pd.api.types.is_datetime64_any_dtype(series):
                return self._validate_dates(non_null_series)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Column validation failed for {column_name}: {str(e)}")
            return False
    
    def _validate_emails(self, series: pd.Series) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_emails = 0
        
        for email in series:
            if isinstance(email, str) and not re.match(email_pattern, email):
                invalid_emails += 1
        
        if invalid_emails > len(series) * 0.1:  # More than 10% invalid
            self.logger.warning(f"High number of invalid emails: {invalid_emails}/{len(series)}")
            return False
        
        return True
    
    def _validate_phones(self, series: pd.Series) -> bool:
        """Validate phone number format"""
        # Basic phone validation - digits, spaces, hyphens, parentheses, plus
        phone_pattern = r'^[\+]?[1-9]?[\d\s\-\(\)]{7,15}$'
        invalid_phones = 0
        
        for phone in series:
            if isinstance(phone, str) and not re.match(phone_pattern, phone.replace(' ', '')):
                invalid_phones += 1
        
        if invalid_phones > len(series) * 0.1:
            self.logger.warning(f"High number of invalid phones: {invalid_phones}/{len(series)}")
            return False
        
        return True
    
    def _validate_dates(self, series: pd.Series) -> bool:
        """Validate date consistency"""
        try:
            # Convert to datetime if string
            if series.dtype == 'object':
                pd.to_datetime(series, errors='raise')
            return True
        except:
            self.logger.warning("Invalid date format detected")
            return False


class RelationshipEnforcer:
    """Enforces logical relationships between columns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.country_data = self._load_country_data()
    
    def _load_country_data(self) -> Dict[str, Dict]:
        """Load country-specific data for relationships"""
        return {
            'USA': {
                'cities': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'],
                'phone_prefix': '+1',
                'currency': 'USD'
            },
            'Canada': {
                'cities': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Ottawa', 'Edmonton'],
                'phone_prefix': '+1',
                'currency': 'CAD'
            },
            'UK': {
                'cities': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Liverpool'],
                'phone_prefix': '+44',
                'currency': 'GBP'
            },
            'Germany': {
                'cities': ['Berlin', 'Munich', 'Hamburg', 'Cologne', 'Frankfurt', 'Stuttgart'],
                'phone_prefix': '+49',
                'currency': 'EUR'
            },
            'France': {
                'cities': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes'],
                'phone_prefix': '+33',
                'currency': 'EUR'
            }
        }
    
    def enforce_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce logical relationships across columns"""
        df_result = df.copy()
        
        try:
            # Date relationships
            df_result = self._enforce_date_relationships(df_result)
            
            # Geographic relationships
            df_result = self._enforce_geographic_relationships(df_result)
            
            # Age-related relationships
            df_result = self._enforce_age_relationships(df_result)
            
            # Email-name relationships
            df_result = self._enforce_email_name_relationships(df_result)
            
            # Job-income relationships
            df_result = self._enforce_job_income_relationships(df_result)
            
            return df_result
            
        except Exception as e:
            self.logger.warning(f"Failed to enforce relationships: {str(e)}")
            return df
    
    def _enforce_date_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce date-based relationships"""
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        # signup_date <= last_login
        if 'signup_date' in df.columns and 'last_login' in df.columns:
            for idx in df.index:
                signup = pd.to_datetime(df.loc[idx, 'signup_date'], errors='coerce')
                last_login = pd.to_datetime(df.loc[idx, 'last_login'], errors='coerce')
                
                if pd.notna(signup) and pd.notna(last_login) and signup > last_login:
                    # Adjust last_login to be after signup_date
                    days_diff = np.random.randint(0, 30)
                    df.loc[idx, 'last_login'] = signup + timedelta(days=days_diff)
        
        # hire_date should be reasonable relative to age
        if 'hire_date' in df.columns and 'age' in df.columns:
            for idx in df.index:
                age = df.loc[idx, 'age']
                if pd.notna(age) and age >= 18:
                    # Hire date should be within working years
                    max_work_years = min(age - 18, 45)
                    years_ago = np.random.randint(0, max_work_years + 1)
                    hire_date = datetime.now() - timedelta(days=years_ago * 365)
                    df.loc[idx, 'hire_date'] = hire_date
        
        return df
    
    def _enforce_geographic_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce country-city-phone relationships"""
        if 'country' not in df.columns:
            return df
        
        # Update cities based on country
        if 'city' in df.columns:
            for idx in df.index:
                country = df.loc[idx, 'country']
                if country in self.country_data:
                    cities = self.country_data[country]['cities']
                    df.loc[idx, 'city'] = np.random.choice(cities)
        
        # Update phone prefixes based on country
        phone_columns = [col for col in df.columns if 'phone' in col.lower()]
        for phone_col in phone_columns:
            for idx in df.index:
                country = df.loc[idx, 'country']
                if country in self.country_data:
                    prefix = self.country_data[country]['phone_prefix']
                    # Generate a simple phone number with correct prefix
                    phone_number = f"{prefix}-{np.random.randint(100, 999)}-{np.random.randint(1000000, 9999999)}"
                    df.loc[idx, phone_col] = phone_number
        
        return df
    
    def _enforce_age_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce age-related relationships"""
        if 'age' not in df.columns:
            return df
        
        # Age affects salary/income
        salary_columns = [col for col in df.columns if any(term in col.lower() for term in ['salary', 'income', 'wage'])]
        for salary_col in salary_columns:
            for idx in df.index:
                age = df.loc[idx, 'age']
                if pd.notna(age):
                    # Base salary increases with age (experience)
                    base_salary = 30000 + (age - 22) * 2000
                    variation = np.random.normal(0, 10000)
                    df.loc[idx, salary_col] = int(max(25000, base_salary + variation))
        
        # Age affects employment status
        if 'employment_status' in df.columns:
            for idx in df.index:
                age = df.loc[idx, 'age']
                if pd.notna(age):
                    if age < 18:
                        df.loc[idx, 'employment_status'] = 'Student'
                    elif age > 65:
                        df.loc[idx, 'employment_status'] = 'Retired'
                    else:
                        status_options = ['Employed', 'Unemployed', 'Self-employed']
                        weights = [0.8, 0.1, 0.1]  # Most people employed
                        df.loc[idx, 'employment_status'] = np.random.choice(status_options, p=weights)
        
        return df
    
    def _enforce_email_name_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create emails from names when possible"""
        name_columns = [col for col in df.columns if 'name' in col.lower()]
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        
        # Check for first_name and last_name columns specifically
        if 'first_name' in df.columns and 'last_name' in df.columns and email_columns:
            email_col = email_columns[0]
            
            for idx in df.index:
                first_name = df.loc[idx, 'first_name']
                last_name = df.loc[idx, 'last_name']
                
                if pd.notna(first_name) and pd.notna(last_name):
                    # Create email from names
                    first_clean = re.sub(r'[^a-zA-Z]', '', str(first_name).lower())
                    last_clean = re.sub(r'[^a-zA-Z]', '', str(last_name).lower())
                    
                    if first_clean and last_clean:
                        email_formats = [
                            f"{first_clean}.{last_clean}@company.com",
                            f"{first_clean}{last_clean}@company.com",
                            f"{first_clean[0]}{last_clean}@company.com"
                        ]
                        df.loc[idx, email_col] = np.random.choice(email_formats)
        
        elif name_columns and email_columns:
            name_col = name_columns[0]
            email_col = email_columns[0]
            
            for idx in df.index:
                name = df.loc[idx, name_col]
                if pd.notna(name) and isinstance(name, str):
                    # Create email from name
                    clean_name = re.sub(r'[^a-zA-Z\s]', '', name.lower())
                    parts = clean_name.split()
                    if len(parts) >= 2:
                        email = f"{parts[0]}.{parts[-1]}@company.com"
                        df.loc[idx, email_col] = email
        
        return df
    
    def _enforce_job_income_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce job title and income relationships"""
        job_columns = [col for col in df.columns if any(term in col.lower() for term in ['job', 'title', 'position', 'role'])]
        salary_columns = [col for col in df.columns if any(term in col.lower() for term in ['salary', 'income', 'wage'])]
        
        if not job_columns or not salary_columns:
            return df
        
        job_col = job_columns[0]
        salary_col = salary_columns[0]
        
        # Job title to salary mapping
        job_salary_map = {
            'ceo': (200000, 500000),
            'manager': (80000, 150000),
            'director': (120000, 250000),
            'engineer': (70000, 130000),
            'developer': (65000, 120000),
            'analyst': (50000, 90000),
            'consultant': (60000, 120000),
            'intern': (30000, 45000),
            'assistant': (35000, 55000),
            'coordinator': (40000, 65000)
        }
        
        for idx in df.index:
            job_title = df.loc[idx, job_col]
            if pd.notna(job_title) and isinstance(job_title, str):
                job_lower = job_title.lower()
                salary_range = None
                
                # Find matching job category
                for key, range_val in job_salary_map.items():
                    if key in job_lower:
                        salary_range = range_val
                        break
                
                if salary_range:
                    # Generate salary within range
                    min_sal, max_sal = salary_range
                    salary = np.random.randint(min_sal, max_sal + 1)
                    df.loc[idx, salary_col] = salary
        
        return df
