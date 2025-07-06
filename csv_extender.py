#!/usr/bin/env python3
"""
📄 CSV Extender Module
Handles CSV file extension with intelligent schema preservation
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from interactive_mode import InteractiveGenerator, ColumnDefinition


class CSVExtender:
    """Handles extending CSV files while preserving schema"""
    
    def __init__(self):
        self.generator = InteractiveGenerator()
    
    def extend_csv_interactive(self):
        """Interactive CSV extension with user prompts"""
        try:
            print("\n📄 CSV EXTENSION MODE")
            print("Upload your CSV and extend it to any size while keeping the exact same schema!")
            print("=" * 60)
            
            # Get CSV file path
            print("\n🔹 Step 1: Specify your CSV file")
            csv_path = input("Enter path to your CSV file (or drag & drop): ").strip().strip('"\'')
            
            if not os.path.exists(csv_path):
                print(f"❌ File not found: {csv_path}")
                return None
            
            # Load and analyze the CSV
            try:
                original_df = pd.read_csv(csv_path)
                print(f"\n✅ CSV loaded successfully!")
                print(f"📊 Original file: {len(original_df)} rows, {len(original_df.columns)} columns")
                print(f"📋 Columns: {list(original_df.columns)}")
                
                # Show sample
                print(f"\n🔍 Sample data (first 3 rows):")
                print(original_df.head(3).to_string(index=False))
                
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
                return None
            
            # Get target size
            print(f"\n🔹 Step 2: Choose target size")
            print(f"Current size: {len(original_df)} rows")
            
            try:
                target_rows = input(f"How many rows do you want in total? (minimum {len(original_df)}): ").strip()
                target_rows = int(target_rows)
                
                if target_rows < len(original_df):
                    print(f"❌ Target must be at least {len(original_df)} (current size)")
                    return None
                elif target_rows == len(original_df):
                    print("💡 No extension needed - file is already the target size!")
                    return None
                    
            except ValueError:
                print("❌ Please enter a valid number")
                return None
            
            # Extend the CSV
            result_path = self.extend_csv_file(csv_path, target_rows)
            return result_path
            
        except KeyboardInterrupt:
            print("\n⏹️ CSV extension cancelled")
            return None
        except Exception as e:
            print(f"❌ Error during CSV extension: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extend_csv_file(self, csv_path, target_rows):
        """Extend a CSV file to target number of rows"""
        try:
            # Load original data
            original_df = pd.read_csv(csv_path)
            
            if target_rows <= len(original_df):
                print("💡 Target size is not larger than current size")
                return None
            
            additional_rows = target_rows - len(original_df)
            print(f"\n🔹 Step 3: Generate {additional_rows} additional rows")
            print(f"🧠 Analyzing your data patterns...")
            
            # Analyze schema and create column definitions
            column_definitions = self._analyze_csv_schema(original_df)
            print(f"🔧 Analyzed {len(column_definitions)} columns")
            
            # Generate additional data
            self.generator.column_definitions = column_definitions
            print(f"🚀 Generating {additional_rows} additional rows...")
            additional_df = self.generator._generate_synthetic_data(additional_rows)
            
            if additional_df is None:
                print("❌ Failed to generate additional data")
                return None
            
            # Apply relationships
            additional_df = self.generator.relationship_enforcer.enforce_relationships(additional_df)
            
            # Combine original and new data
            extended_df = pd.concat([original_df, additional_df], ignore_index=True)
            
            # Save the extended CSV
            output_path = self._save_extended_csv(csv_path, extended_df, target_rows)
            
            print(f"\n🎉 SUCCESS! CSV extended successfully!")
            print(f"📊 Original: {len(original_df)} rows")
            print(f"📊 Extended: {len(extended_df)} rows (+{additional_rows} new)")
            print(f"💾 Saved as: {output_path}")
            
            # Show sample of new data
            print(f"\n🔍 Sample of new data (last 3 rows):")
            print(extended_df.tail(3).to_string(index=False))
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error extending CSV: {e}")
            return None
    
    def _analyze_csv_schema(self, df):
        """Analyze CSV schema and create appropriate column definitions"""
        column_definitions = []
        
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            
            # Detect column type and constraints
            if col_dtype in ['int64', 'int32']:
                min_val = int(df[col].min())
                max_val = int(df[col].max())
                column_definitions.append(ColumnDefinition(col, 'integer', min_value=min_val, max_value=max_val))
            elif col_dtype in ['float64', 'float32']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                column_definitions.append(ColumnDefinition(col, 'float', min_value=min_val, max_value=max_val))
            elif any(keyword in col.lower() for keyword in ['name', 'first', 'last']):
                column_definitions.append(ColumnDefinition(col, 'name'))
            elif any(keyword in col.lower() for keyword in ['email', 'mail']):
                column_definitions.append(ColumnDefinition(col, 'email'))
            elif any(keyword in col.lower() for keyword in ['phone', 'tel']):
                column_definitions.append(ColumnDefinition(col, 'phone'))
            elif any(keyword in col.lower() for keyword in ['address', 'street', 'city']):
                column_definitions.append(ColumnDefinition(col, 'address'))
            elif any(keyword in col.lower() for keyword in ['date', 'time']):
                column_definitions.append(ColumnDefinition(col, 'date'))
            elif any(keyword in col.lower() for keyword in ['id', 'uuid', 'key']):
                column_definitions.append(ColumnDefinition(col, 'uuid', uuid_format='short'))
            elif df[col].nunique() < 10 and col_dtype == 'object':  # Small categorical
                categories = df[col].unique().tolist()
                column_definitions.append(ColumnDefinition(col, 'categorical', categories=categories))
            elif col_dtype == 'object':  # Text data that's not clearly categorical
                # Check if it looks like names (has spaces, proper case, etc.)
                sample_values = df[col].dropna().head(5).tolist()
                looks_like_names = any(
                    isinstance(val, str) and ' ' in val and val.replace(' ', '').isalpha() 
                    for val in sample_values
                )
                if looks_like_names:
                    column_definitions.append(ColumnDefinition(col, 'name'))
                else:
                    column_definitions.append(ColumnDefinition(col, 'text'))
            else:
                column_definitions.append(ColumnDefinition(col, 'text'))
        
        return column_definitions
    
    def _save_extended_csv(self, original_path, extended_df, target_rows):
        """Save the extended CSV with a descriptive filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_filename = f"{base_name}_extended_{target_rows}rows_{timestamp}.csv"
        output_path = Path('output') / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        extended_df.to_csv(output_path, index=False)
        return output_path


def main():
    """For testing the CSV extender directly"""
    extender = CSVExtender()
    result = extender.extend_csv_interactive()
    if result:
        print(f"\n✅ CSV extension completed: {result}")
    else:
        print("\n❌ CSV extension failed or cancelled")


if __name__ == "__main__":
    main()
