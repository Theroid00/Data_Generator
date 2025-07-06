"""
Interactive Mode Module
Handles generation of synthetic data from scratch via interactive prompts
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from faker import Faker

from logic_rules import DataValidator, RelationshipEnforcer


class ColumnDefinition:
    """Represents a column definition with type and constraints"""
    
    def __init__(self, name: str, data_type: str, **kwargs):
        self.name = name
        self.data_type = data_type
        self.min_value = kwargs.get('min_value')
        self.max_value = kwargs.get('max_value')
        self.categories = kwargs.get('categories', [])
        self.distribution = kwargs.get('distribution', 'uniform')
        self.depends_on = kwargs.get('depends_on')
        self.relationship = kwargs.get('relationship')
        self.format_pattern = kwargs.get('format_pattern')
        # UUID specific attributes
        self.uuid_format = kwargs.get('uuid_format', 'short')
        self.uuid_prefix = kwargs.get('uuid_prefix', '')
        self.uuid_length = kwargs.get('uuid_length', 8)


class InteractiveGenerator:
    """Handles interactive generation of synthetic data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.faker = Faker()
        self.validator = DataValidator()
        self.relationship_enforcer = RelationshipEnforcer()
        self.column_definitions = []
    
    def run_interactive_mode(self) -> Optional[Path]:
        """Run the interactive data generation process"""
        print("\n🔹 Interactive Synthetic Data Generation")
        print("-" * 50)
        
        try:
            # Get basic parameters
            num_rows = self._get_num_rows()
            num_columns = self._get_num_columns()
            
            # Define columns
            self.column_definitions = []
            for i in range(num_columns):
                print(f"\n📋 Column {i + 1}:")
                column_def = self._define_column()
                self.column_definitions.append(column_def)
            
            # Ask about saving schema
            if self._ask_yes_no("\nWould you like to save this schema for reuse?"):
                self._save_schema()
            
            # Generate data
            print(f"\n🔹 Generating {num_rows} rows of synthetic data...")
            df = self._generate_synthetic_data(num_rows)
            
            if df is not None:
                # Apply relationships
                df = self.relationship_enforcer.enforce_relationships(df)
                
                # Validate data
                if self.validator.validate_dataframe(df):
                    print("✅ Data validation passed")
                else:
                    print("⚠️ Data validation warnings (see logs)")
                
                # Save data
                output_path = self._save_generated_data(df)
                return output_path
            else:
                print("❌ Failed to generate data")
                return None
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Generation cancelled by user")
            return None
        except Exception as e:
            self.logger.error(f"Interactive generation failed: {str(e)}")
            print(f"❌ Error: {str(e)}")
            return None
    
    def _get_num_rows(self) -> int:
        """Get number of rows to generate"""
        while True:
            try:
                num_rows = int(input("How many rows to generate? "))
                if num_rows > 0:
                    return num_rows
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _get_num_columns(self) -> int:
        """Get number of columns to generate"""
        while True:
            try:
                num_columns = int(input("How many columns? "))
                if num_columns > 0:
                    return num_columns
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    def _define_column(self) -> ColumnDefinition:
        """Define a single column interactively"""
        # Get column name
        name = input("Column name: ").strip()
        
        # Get data type
        print("\nAvailable data types:")
        print("[1] name - Full names")
        print("[2] email - Email addresses")
        print("[3] integer - Whole numbers")
        print("[4] float - Decimal numbers")
        print("[5] date - Dates")
        print("[6] categorical - Fixed categories")
        print("[7] phone - Phone numbers")
        print("[8] address - Addresses")
        print("[9] text - Free text")
        print("[10] boolean - True/False")
        print("[11] uuid - UUID primary keys (universally unique identifiers)")
        
        type_map = {
            1: 'name', 2: 'email', 3: 'integer', 4: 'float', 5: 'date',
            6: 'categorical', 7: 'phone', 8: 'address', 9: 'text', 10: 'boolean', 11: 'uuid'
        }
        
        while True:
            try:
                choice = int(input("Select type (1-11): "))
                if choice in type_map:
                    data_type = type_map[choice]
                    break
                else:
                    print("Please enter a number between 1 and 11.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get type-specific parameters
        kwargs = self._get_type_specific_params(data_type)
        
        return ColumnDefinition(name, data_type, **kwargs)
    
    def _get_type_specific_params(self, data_type: str) -> Dict[str, Any]:
        """Get type-specific parameters for column definition"""
        kwargs = {}
        
        if data_type == 'integer':
            kwargs['min_value'] = self._get_optional_int("Minimum value (optional): ")
            kwargs['max_value'] = self._get_optional_int("Maximum value (optional): ")
            kwargs['distribution'] = self._get_distribution()
        
        elif data_type == 'float':
            kwargs['min_value'] = self._get_optional_float("Minimum value (optional): ")
            kwargs['max_value'] = self._get_optional_float("Maximum value (optional): ")
            kwargs['distribution'] = self._get_distribution()
        
        elif data_type == 'categorical':
            categories = []
            print("Enter categories (one per line, empty line to finish):")
            while True:
                category = input("  Category: ").strip()
                if not category:
                    break
                categories.append(category)
            kwargs['categories'] = categories
        
        elif data_type == 'uuid':
            print("UUID format options:")
            print("[1] Short ID (8 chars): e.g., A7D3E41B")
            print("[2] Medium ID (12 chars): e.g., A7D3-E41B-9F2C")
            print("[3] Readable ID: e.g., USER-2024-001234")
            print("[4] Sequential ID: e.g., ID000001, ID000002")
            print("[5] Custom format with prefix")
            
            while True:
                try:
                    uuid_choice = int(input("Select ID format (1-5): "))
                    if uuid_choice == 1:
                        kwargs['uuid_format'] = 'short'
                        break
                    elif uuid_choice == 2:
                        kwargs['uuid_format'] = 'medium'
                        break
                    elif uuid_choice == 3:
                        kwargs['uuid_format'] = 'readable'
                        prefix = input("Enter prefix (e.g., USER, CUST): ").strip()
                        kwargs['uuid_prefix'] = prefix if prefix else 'ID'
                        break
                    elif uuid_choice == 4:
                        kwargs['uuid_format'] = 'sequential'
                        prefix = input("Enter prefix (e.g., ID, REF): ").strip()
                        kwargs['uuid_prefix'] = prefix if prefix else 'ID'
                        break
                    elif uuid_choice == 5:
                        kwargs['uuid_format'] = 'custom'
                        prefix = input("Enter prefix: ").strip()
                        kwargs['uuid_prefix'] = prefix
                        length = self._get_optional_int("Enter ID length (4-16): ")
                        kwargs['uuid_length'] = max(4, min(16, length if length else 8))
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\nOperation cancelled. Using default UUID format.")
                    kwargs['uuid_format'] = 'short'
                    break

        return kwargs
    
    def _get_distribution(self) -> str:
        """Get distribution type with error handling"""
        print("Distribution options:")
        print("[1] Uniform (default)")
        print("[2] Normal")
        print("[3] Random")

        try:
            choice = input("Select distribution (1-3, default 1): ").strip()
            if choice == '2':
                return 'normal'
            elif choice == '3':
                return 'random'
            else:
                return 'uniform'
        except (ValueError, KeyboardInterrupt):
            return 'uniform'

    def _get_optional_int(self, prompt: str) -> Optional[int]:
        """Get optional integer input with error handling"""
        try:
            value = input(prompt).strip()
            return int(value) if value else None
        except ValueError:
            print("Invalid number, using default")
            return None
        except KeyboardInterrupt:
            return None

    def _get_optional_float(self, prompt: str) -> Optional[float]:
        """Get optional float input with error handling"""
        try:
            value = input(prompt).strip()
            return float(value) if value else None
        except ValueError:
            print("Invalid number, using default")
            return None
        except KeyboardInterrupt:
            return None

    def _ask_yes_no(self, prompt: str) -> bool:
        """Ask yes/no question with error handling"""
        try:
            while True:
                response = input(f"{prompt} (y/n): ").strip().lower()
                if response in ['y', 'yes', 'true', '1']:
                    return True
                elif response in ['n', 'no', 'false', '0']:
                    return False
                else:
                    print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            return False

    def _save_schema(self):
        """Save column schema for reuse with error handling"""
        try:
            schema_name = input("Schema name: ").strip()
            if not schema_name:
                print("No schema name provided, skipping save")
                return

            schema_data = {
                'name': schema_name,
                'created_at': datetime.now().isoformat(),
                'columns': []
            }
            
            for col_def in self.column_definitions:
                col_data = {
                    'name': col_def.name,
                    'data_type': col_def.data_type,
                    'min_value': col_def.min_value,
                    'max_value': col_def.max_value,
                    'categories': col_def.categories,
                    'distribution': col_def.distribution,
                    'uuid_format': col_def.uuid_format,
                    'uuid_prefix': col_def.uuid_prefix,
                    'uuid_length': col_def.uuid_length
                }
                schema_data['columns'].append(col_data)
            
            # Save to templates directory
            templates_dir = Path('templates')
            templates_dir.mkdir(exist_ok=True)

            schema_path = templates_dir / f"custom_{schema_name}.json"
            with open(schema_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
            
            print(f"✅ Schema saved: {schema_path}")

        except Exception as e:
            print(f"❌ Failed to save schema: {e}")

    def _generate_synthetic_data(self, num_rows: int) -> Optional[pd.DataFrame]:
        """Generate synthetic data from column definitions with comprehensive error handling"""
        try:
            if not self.column_definitions:
                self.logger.error("No column definitions provided")
                return None

            data = {}
            
            # Generate each column with individual error handling
            for col_def in self.column_definitions:
                try:
                    column_data = self._generate_column_data(col_def, num_rows)
                    if column_data is not None:
                        data[col_def.name] = column_data
                    else:
                        self.logger.warning(f"Failed to generate data for column {col_def.name}")
                        # Create fallback data
                        data[col_def.name] = [f"Error_{i}" for i in range(num_rows)]

                except Exception as e:
                    self.logger.error(f"Error generating column {col_def.name}: {e}")
                    # Create fallback data
                    data[col_def.name] = [f"Error_{i}" for i in range(num_rows)]

            if not data:
                self.logger.error("No data generated for any columns")
                return None

            # Create DataFrame with error handling
            try:
                df = pd.DataFrame(data)

                # Validate DataFrame
                if df.empty:
                    self.logger.error("Generated DataFrame is empty")
                    return None

                if len(df) != num_rows:
                    self.logger.warning(f"Expected {num_rows} rows, got {len(df)}")

                return df

            except Exception as e:
                self.logger.error(f"Failed to create DataFrame: {e}")
                return None

        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {e}")
            return None
    
    def _generate_column_data(self, col_def: ColumnDefinition, num_rows: int) -> Optional[List]:
        """Generate data for a single column with comprehensive error handling"""
        try:
            data_type = col_def.data_type

            if data_type == 'integer':
                return self._generate_integer_data(col_def, num_rows)
            elif data_type == 'float':
                return self._generate_float_data(col_def, num_rows)
            elif data_type == 'categorical':
                return self._generate_categorical_data(col_def, num_rows)
            elif data_type == 'name':
                return self._generate_name_data(col_def, num_rows)
            elif data_type == 'email':
                return self._generate_email_data(col_def, num_rows)
            elif data_type == 'phone':
                return self._generate_phone_data(col_def, num_rows)
            elif data_type == 'address':
                return self._generate_address_data(col_def, num_rows)
            elif data_type == 'date':
                return self._generate_date_data(col_def, num_rows)
            elif data_type == 'text':
                return self._generate_text_data(col_def, num_rows)
            elif data_type == 'boolean':
                return self._generate_boolean_data(col_def, num_rows)
            elif data_type == 'uuid':
                return self._generate_uuid_data(col_def, num_rows)
            else:
                self.logger.warning(f"Unknown data type: {data_type}")
                return [f"Unknown_{i}" for i in range(num_rows)]

        except Exception as e:
            self.logger.error(f"Failed to generate {data_type} data: {e}")
            return None

    def _generate_integer_data(self, col_def: ColumnDefinition, num_rows: int) -> List[int]:
        """Generate integer data with error handling"""
        try:
            min_val = col_def.min_value if col_def.min_value is not None else 1
            max_val = col_def.max_value if col_def.max_value is not None else 1000

            # Ensure valid range
            if min_val >= max_val:
                max_val = min_val + 100

            if col_def.distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                data = np.random.normal(mean, std, num_rows)
                return np.clip(data, min_val, max_val).round().astype(int).tolist()
            else:
                return np.random.randint(min_val, max_val + 1, num_rows).tolist()

        except Exception as e:
            self.logger.error(f"Integer generation failed: {e}")
            return list(range(1, num_rows + 1))

    def _generate_float_data(self, col_def: ColumnDefinition, num_rows: int) -> List[float]:
        """Generate float data with error handling"""
        try:
            min_val = col_def.min_value if col_def.min_value is not None else 0.0
            max_val = col_def.max_value if col_def.max_value is not None else 1.0

            # Ensure valid range
            if min_val >= max_val:
                max_val = min_val + 1.0

            if col_def.distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                data = np.random.normal(mean, std, num_rows)
                return np.clip(data, min_val, max_val).round(2).tolist()
            else:
                return (np.random.uniform(min_val, max_val, num_rows).round(2)).tolist()

        except Exception as e:
            self.logger.error(f"Float generation failed: {e}")
            return [0.0] * num_rows

    def _generate_categorical_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate categorical data with error handling"""
        try:
            categories = col_def.categories
            if not categories:
                categories = ['Category A', 'Category B', 'Category C']

            return np.random.choice(categories, num_rows).tolist()

        except Exception as e:
            self.logger.error(f"Categorical generation failed: {e}")
            return ['Default Category'] * num_rows

    def _generate_name_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate name data with error handling"""
        try:
            name_type = col_def.name.lower()
            if 'first' in name_type:
                return [self.faker.first_name() for _ in range(num_rows)]
            elif 'last' in name_type:
                return [self.faker.last_name() for _ in range(num_rows)]
            else:
                return [self.faker.name() for _ in range(num_rows)]

        except Exception as e:
            self.logger.error(f"Name generation failed: {e}")
            return [f"Name_{i}" for i in range(num_rows)]

    def _generate_email_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate email data with error handling"""
        try:
            return [self.faker.email() for _ in range(num_rows)]
        except Exception as e:
            self.logger.error(f"Email generation failed: {e}")
            return [f"user{i}@example.com" for i in range(num_rows)]

    def _generate_phone_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate phone data with error handling"""
        try:
            return [self.faker.phone_number() for _ in range(num_rows)]
        except Exception as e:
            self.logger.error(f"Phone generation failed: {e}")
            return [f"+1-555-000-{str(i).zfill(4)}" for i in range(num_rows)]

    def _generate_address_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate address data with error handling"""
        try:
            return [self.faker.address().replace('\n', ', ') for _ in range(num_rows)]
        except Exception as e:
            self.logger.error(f"Address generation failed: {e}")
            return [f"{i} Main Street, City, State" for i in range(1, num_rows + 1)]

    def _generate_date_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate date data with error handling"""
        try:
            dates = []
            for _ in range(num_rows):
                date = self.faker.date_between(start_date='-10y', end_date='today')
                dates.append(date.strftime('%Y-%m-%d'))
            return dates
        except Exception as e:
            self.logger.error(f"Date generation failed: {e}")
            return [datetime.now().strftime('%Y-%m-%d')] * num_rows

    def _generate_text_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate text data with error handling"""
        try:
            return [self.faker.text(max_nb_chars=100) for _ in range(num_rows)]
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return [f"Sample text {i}" for i in range(num_rows)]

    def _generate_boolean_data(self, col_def: ColumnDefinition, num_rows: int) -> List[bool]:
        """Generate boolean data with error handling"""
        try:
            return np.random.choice([True, False], num_rows).tolist()
        except Exception as e:
            self.logger.error(f"Boolean generation failed: {e}")
            return [True] * num_rows

    def _generate_uuid_data(self, col_def: ColumnDefinition, num_rows: int) -> List[str]:
        """Generate UUID data with comprehensive error handling"""
        try:
            import uuid
            import random
            import string

            uuid_format = col_def.uuid_format
            uuid_prefix = col_def.uuid_prefix
            uuid_length = col_def.uuid_length

            uuids = []

            for i in range(num_rows):
                try:
                    if uuid_format == 'short':
                        # Short alphanumeric IDs: A7D3E41B
                        uuid_val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

                    elif uuid_format == 'medium':
                        # Medium format with dashes: A7D3-E41B-9F2C
                        part1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
                        part2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
                        part3 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
                        uuid_val = f"{part1}-{part2}-{part3}"

                    elif uuid_format == 'readable':
                        # Readable format: USER-2024-001234
                        year = datetime.now().year
                        prefix = uuid_prefix or 'ID'
                        uuid_val = f"{prefix}-{year}-{str(i+1).zfill(6)}"

                    elif uuid_format == 'sequential':
                        # Sequential format: ID000001, ID000002
                        prefix = uuid_prefix or 'ID'
                        uuid_val = f"{prefix}{str(i+1).zfill(6)}"

                    elif uuid_format == 'custom':
                        # Custom length with prefix
                        prefix = uuid_prefix or ''
                        length = max(4, min(16, uuid_length))
                        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
                        uuid_val = f"{prefix}{random_part}"

                    elif uuid_format == 'standard':
                        # Standard UUID4: 550e8400-e29b-41d4-a716-446655440000
                        uuid_val = str(uuid.uuid4())

                    else:
                        # Fallback to short format
                        uuid_val = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

                    uuids.append(uuid_val)

                except Exception as uuid_error:
                    self.logger.warning(f"UUID generation error for row {i}: {uuid_error}")
                    # Fallback UUID
                    uuids.append(f"ID{str(i+1).zfill(6)}")

            # Ensure uniqueness
            if len(set(uuids)) < len(uuids):
                self.logger.warning("Duplicate UUIDs detected, making them unique")
                unique_uuids = []
                seen = set()
                for i, uuid_val in enumerate(uuids):
                    if uuid_val in seen:
                        uuid_val = f"{uuid_val}_{i}"
                    seen.add(uuid_val)
                    unique_uuids.append(uuid_val)
                return unique_uuids

            return uuids

        except Exception as e:
            self.logger.error(f"UUID generation failed: {e}")
            # Fallback to simple sequential IDs
            return [f"ID{str(i+1).zfill(6)}" for i in range(num_rows)]

    def _save_generated_data(self, df: pd.DataFrame) -> Path:
        """Save generated data with error handling"""
        try:
            # Create output directory
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interactive_generated_{timestamp}.csv"
            output_path = output_dir / filename

            # Save DataFrame
            df.to_csv(output_path, index=False)

            print(f"✅ Data saved: {output_path}")
            print(f"📊 Generated {len(df)} rows with {len(df.columns)} columns")

            # Show sample data
            print(f"\n📋 Sample data (first 3 rows):")
            print(df.head(3).to_string(index=False))

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            print(f"❌ Error saving data: {e}")
            return None
