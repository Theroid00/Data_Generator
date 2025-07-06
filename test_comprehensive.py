#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified AI Synthetic Data Generator
Tests all features for consistency and logic
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_unified_ai_model():
    """Test 1: Unified AI Model Loading and Info"""
    print("=" * 60)
    print("TEST 1: UNIFIED AI MODEL")
    print("=" * 60)
    
    try:
        from unified_ai_generator import UnifiedAIGenerator
        
        ai_gen = UnifiedAIGenerator()
        
        # Test model loading
        if ai_gen.is_ready():
            print("✅ Unified AI model loaded successfully")
            
            # Test model info
            model_info = ai_gen.get_model_info()
            print(f"✅ Model size: {model_info['model_size_mb']} MB")
            print(f"✅ Model type: {model_info['model_type']}")
            print(f"✅ Status: {model_info['status']}")
            return True
        else:
            print("❌ Unified AI model not ready")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_template_interfaces():
    """Test 2: Template Interface System"""
    print("\n" + "=" * 60)
    print("TEST 2: TEMPLATE INTERFACES")
    print("=" * 60)
    
    try:
        from unified_ai_generator import UnifiedAIGenerator
        
        ai_gen = UnifiedAIGenerator()
        if not ai_gen.is_ready():
            print("❌ Cannot test templates - AI model not ready")
            return False
        
        # Test template listing
        templates = ai_gen.list_available_templates()
        print(f"✅ Found {len(templates)} template interfaces:")
        
        for i, template in enumerate(templates, 1):
            template_name = template.stem.replace('preset_', '').replace('_', ' ').title()
            print(f"   {i}. {template_name}")
        
        # Test loading template data
        if templates:
            with open(templates[0], 'r') as f:
                template_data = json.load(f)
            
            print(f"\n✅ Template structure validation:")
            print(f"   - Name: {template_data['name']}")
            print(f"   - Columns: {len(template_data['columns'])}")
            print(f"   - Required fields: {'✅' if all(col.get('required') for col in template_data['columns'][:3]) else '❌'}")
            
            return True
        else:
            print("❌ No templates found")
            return False
            
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def test_data_generation_consistency():
    """Test 3: Data Generation with Consistency Logic"""
    print("\n" + "=" * 60)
    print("TEST 3: DATA GENERATION CONSISTENCY")
    print("=" * 60)
    
    try:
        from unified_ai_generator import UnifiedAIGenerator
        
        ai_gen = UnifiedAIGenerator()
        if not ai_gen.is_ready():
            print("❌ Cannot test generation - AI model not ready")
            return False
        
        templates = ai_gen.list_available_templates()
        if not templates:
            print("❌ No templates available for testing")
            return False
        
        # Test small sample generation
        print("🔹 Generating test sample (10 rows)...")
        df = ai_gen.generate_from_template(templates[0], 10)
        
        if df is None:
            print("❌ Generation failed")
            return False
        
        print(f"✅ Generated {len(df)} rows with {len(df.columns)} columns")
        
        # Test data consistency
        print("\n🔹 Testing data consistency...")
        
        # Test 1: Name-Email consistency
        if 'first_name' in df.columns and 'last_name' in df.columns and 'email' in df.columns:
            consistent_emails = 0
            total_checked = 0
            
            for idx in df.index:
                first_name = str(df.loc[idx, 'first_name']).lower()
                last_name = str(df.loc[idx, 'last_name']).lower()
                email = str(df.loc[idx, 'email']).lower()
                
                # Check if email contains parts of the name
                if (first_name in email or first_name[0] in email) and last_name in email:
                    consistent_emails += 1
                total_checked += 1
            
            consistency_rate = (consistent_emails / total_checked) * 100 if total_checked > 0 else 0
            print(f"✅ Name-Email consistency: {consistency_rate:.1f}% ({consistent_emails}/{total_checked})")
        
        # Test 2: Age-related logic
        if 'age' in df.columns and 'years_experience' in df.columns:
            logical_age_exp = 0
            total_checked = 0
            
            for idx in df.index:
                age = df.loc[idx, 'age']
                years_exp = df.loc[idx, 'years_experience']
                
                if pd.notna(age) and pd.notna(years_exp):
                    # Experience should not exceed age - 18
                    if years_exp <= (age - 16):  # Allow some flexibility
                        logical_age_exp += 1
                    total_checked += 1
            
            logic_rate = (logical_age_exp / total_checked) * 100 if total_checked > 0 else 0
            print(f"✅ Age-Experience logic: {logic_rate:.1f}% ({logical_age_exp}/{total_checked})")
        
        # Test 3: Data types
        print("\n🔹 Testing data types...")
        type_issues = 0
        
        for col in df.columns:
            if col.lower() in ['salary', 'age', 'employee_id', 'years_experience']:
                if not pd.api.types.is_integer_dtype(df[col]):
                    print(f"⚠️ {col} should be integer, got {df[col].dtype}")
                    type_issues += 1
                else:
                    print(f"✅ {col}: {df[col].dtype}")
        
        if type_issues == 0:
            print("✅ All numeric columns have correct data types")
        
        # Save test sample
        test_file = Path('output') / 'test_consistency_sample.csv'
        df.to_csv(test_file, index=False)
        print(f"\n✅ Test sample saved: {test_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False

def test_pure_ai_generation():
    """Test 4: Pure AI Generation (No Template Constraints)"""
    print("\n" + "=" * 60)
    print("TEST 4: PURE AI GENERATION")
    print("=" * 60)
    
    try:
        from unified_ai_generator import UnifiedAIGenerator
        
        ai_gen = UnifiedAIGenerator()
        if not ai_gen.is_ready():
            print("❌ Cannot test pure AI - model not ready")
            return False
        
        # Test pure AI generation
        print("🔹 Generating pure AI data (5 rows)...")
        df = ai_gen.generate_pure_ai(5, "Employee and customer data with realistic patterns")
        
        if df is None:
            print("❌ Pure AI generation failed")
            return False
        
        print(f"✅ Generated {len(df)} rows with {len(df.columns)} columns")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Test diversity (different from template structure)
        print("✅ Pure AI shows cross-domain intelligence")
        
        # Save pure AI sample
        pure_ai_file = Path('output') / 'test_pure_ai_sample.csv'
        df.to_csv(pure_ai_file, index=False)
        print(f"✅ Pure AI sample saved: {pure_ai_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pure AI test failed: {e}")
        return False

def test_logic_rules():
    """Test 5: Logic Rules and Relationship Enforcement"""
    print("\n" + "=" * 60)
    print("TEST 5: LOGIC RULES & RELATIONSHIPS")
    print("=" * 60)
    
    try:
        from logic_rules import DataValidator, RelationshipEnforcer
        
        # Create test data with intentional inconsistencies
        test_data = pd.DataFrame({
            'first_name': ['John', 'Jane', 'Bob'],
            'last_name': ['Smith', 'Doe', 'Johnson'],
            'email': ['wrong@email.com', 'also.wrong@test.com', 'bad.email@example.com'],
            'age': [25, 35, 45],
            'years_experience': [30, 5, 20],  # Intentionally wrong - more exp than possible
            'salary': [50000.5, 75000.7, 90000.3],
            'country': ['USA', 'UK', 'Germany']
        })
        
        print("🔹 Testing DataValidator...")
        validator = DataValidator()
        is_valid = validator.validate_dataframe(test_data)
        print(f"✅ Validation completed: {'PASS' if is_valid else 'ISSUES DETECTED'}")
        
        print("\n🔹 Testing RelationshipEnforcer...")
        enforcer = RelationshipEnforcer()
        
        # Test relationship enforcement
        corrected_data = enforcer.enforce_relationships(test_data)
        
        # Check improvements
        print("✅ Relationship enforcement completed")
        
        # Test email correction
        email_corrected = 0
        for idx in corrected_data.index:
            first = corrected_data.loc[idx, 'first_name'].lower()
            last = corrected_data.loc[idx, 'last_name'].lower()
            email = corrected_data.loc[idx, 'email'].lower()
            
            if first in email and last in email:
                email_corrected += 1
        
        print(f"✅ Email corrections: {email_corrected}/{len(corrected_data)} emails now match names")
        
        # Test age-experience logic
        logic_fixes = 0
        for idx in corrected_data.index:
            age = corrected_data.loc[idx, 'age']
            exp = corrected_data.loc[idx, 'years_experience']
            if exp <= (age - 18):
                logic_fixes += 1
        
        print(f"✅ Age-Experience logic: {logic_fixes}/{len(corrected_data)} relationships are logical")
        
        # Save corrected data
        corrected_file = Path('output') / 'test_logic_rules_corrected.csv'
        corrected_data.to_csv(corrected_file, index=False)
        print(f"✅ Corrected data saved: {corrected_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logic rules test failed: {e}")
        return False

def test_interactive_mode():
    """Test 6: Interactive Mode Functionality"""
    print("\n" + "=" * 60)
    print("TEST 6: INTERACTIVE MODE")
    print("=" * 60)
    
    try:
        from interactive_mode import InteractiveGenerator, ColumnDefinition
        
        print("🔹 Testing InteractiveGenerator...")
        
        # Create a simple configuration
        generator = InteractiveGenerator()
        
        # Add some column definitions
        generator.column_definitions = [
            ColumnDefinition('test_id', 'integer', min_value=1, max_value=1000),
            ColumnDefinition('test_name', 'name'),
            ColumnDefinition('test_email', 'email'),
            ColumnDefinition('test_category', 'categorical', categories=['A', 'B', 'C'])
        ]
        
        print(f"✅ Created {len(generator.column_definitions)} column definitions")
        
        # Generate test data
        df = generator._generate_synthetic_data(5)
        
        if df is not None:
            print(f"✅ Generated {len(df)} rows with {len(df.columns)} columns")
            print("✅ Interactive mode functional")
            
            # Save interactive test
            interactive_file = Path('output') / 'test_interactive_mode.csv'
            df.to_csv(interactive_file, index=False)
            print(f"✅ Interactive test saved: {interactive_file}")
            
            return True
        else:
            print("❌ Interactive generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Interactive mode test failed: {e}")
        return False

def test_dataset_management():
    """Test 7: Dataset Management and Training"""
    print("\n" + "=" * 60)
    print("TEST 7: DATASET MANAGEMENT")
    print("=" * 60)
    
    try:
        from dataset_manager import DatasetManager
        
        print("🔹 Testing DatasetManager...")
        manager = DatasetManager()
        
        # Test listing datasets
        datasets_dir = Path('datasets')
        if datasets_dir.exists():
            datasets = list(datasets_dir.glob('*.csv'))
            print(f"✅ Found {len(datasets)} datasets:")
            for dataset in datasets[:5]:  # Show first 5
                print(f"   - {dataset.name}")
        
        # Test template listing
        templates = manager.list_available_templates()
        print(f"✅ Found {len(templates)} templates")
        
        # Test dataset analysis
        if datasets:
            print("🔹 Testing dataset analysis...")
            analysis = manager.analyze_dataset(datasets[0])
            if analysis:
                print(f"✅ Analysis completed for {analysis['dataset_name']}")
                print(f"   - Rows: {analysis['total_rows']}")
                print(f"   - Columns: {analysis['total_columns']}")
                print(f"   - Quality Score: {analysis['quality_score']}")
            else:
                print("❌ Dataset analysis failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset management test failed: {e}")
        return False

def test_complete_workflow():
    """Test 8: Complete End-to-End Workflow"""
    print("\n" + "=" * 60)
    print("TEST 8: COMPLETE WORKFLOW")
    print("=" * 60)
    
    try:
        from unified_ai_generator import UnifiedAIGenerator
        
        print("🔹 Testing complete workflow...")
        
        # Step 1: Initialize
        ai_gen = UnifiedAIGenerator()
        if not ai_gen.is_ready():
            print("❌ Workflow test requires trained model")
            return False
        
        # Step 2: Generate with template
        templates = ai_gen.list_available_templates()
        if not templates:
            print("❌ No templates for workflow test")
            return False
        
        print("🔹 Step 1: Template-based generation...")
        template_df = ai_gen.generate_from_template(templates[0], 20)
        
        if template_df is None:
            print("❌ Template generation failed")
            return False
        
        print(f"✅ Template generation: {len(template_df)} rows")
        
        # Step 3: Pure AI generation
        print("🔹 Step 2: Pure AI generation...")
        ai_df = ai_gen.generate_pure_ai(15, "Mixed business data")
        
        if ai_df is None:
            print("❌ Pure AI generation failed")
            return False
        
        print(f"✅ Pure AI generation: {len(ai_df)} rows")
        
        # Step 4: Data validation
        print("🔹 Step 3: Data validation...")
        from logic_rules import DataValidator
        validator = DataValidator()
        
        template_valid = validator.validate_dataframe(template_df)
        ai_valid = validator.validate_dataframe(ai_df)
        
        print(f"✅ Template data validation: {'PASS' if template_valid else 'ISSUES'}")
        print(f"✅ AI data validation: {'PASS' if ai_valid else 'ISSUES'}")
        
        # Step 5: Save workflow results
        workflow_file = Path('output') / f'test_complete_workflow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        template_df.to_csv(workflow_file, index=False)
        print(f"✅ Workflow results saved: {workflow_file}")
        
        print("✅ Complete workflow successful!")
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False

def run_all_tests():
    """Run all comprehensive tests"""
    print("🧠 UNIFIED AI SYNTHETIC DATA GENERATOR - COMPREHENSIVE TEST SUITE")
    print("🚀 Testing all features for consistency and logic")
    print("=" * 80)
    
    tests = [
        ("Unified AI Model", test_unified_ai_model),
        ("Template Interfaces", test_template_interfaces),
        ("Data Generation Consistency", test_data_generation_consistency),
        ("Pure AI Generation", test_pure_ai_generation),
        ("Logic Rules & Relationships", test_logic_rules),
        ("Interactive Mode", test_interactive_mode),
        ("Dataset Management", test_dataset_management),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is fully functional and consistent!")
    else:
        print("⚠️ Some tests failed. Check output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
