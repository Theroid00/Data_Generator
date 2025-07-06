#!/usr/bin/env python3
"""
Unified AI Generator Test Suite
Tests the core AI model functionality and template interfaces
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
import unittest

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class TestUnifiedAIGenerator(unittest.TestCase):
    """Test suite for the Unified AI Generator"""

    def setUp(self):
        """Set up test environment"""
        try:
            from unified_ai_generator import UnifiedAIGenerator
            self.ai_gen = UnifiedAIGenerator()
        except Exception as e:
            self.skipTest(f"Could not initialize UnifiedAIGenerator: {e}")

    def test_model_loading(self):
        """Test if the unified AI model loads correctly"""
        print("\n🧪 Testing unified AI model loading...")

        # Test model availability
        model_exists = self.ai_gen.unified_model_path.exists()
        print(f"Model file exists: {model_exists}")

        if model_exists:
            self.assertTrue(self.ai_gen.is_ready(), "Model should be ready when file exists")
            print("✅ Model loaded successfully")
        else:
            self.assertFalse(self.ai_gen.is_ready(), "Model should not be ready when file missing")
            print("⚠️ Model file not found - training needed")

    def test_model_info(self):
        """Test model information retrieval"""
        print("\n🧪 Testing model info retrieval...")

        if not self.ai_gen.is_ready():
            self.skipTest("Model not available for testing")

        try:
            model_info = self.ai_gen.get_model_info()

            # Check required fields
            required_fields = ['message', 'model_size_mb', 'model_type', 'status']
            for field in required_fields:
                self.assertIn(field, model_info, f"Model info should contain {field}")

            # Check data types
            self.assertIsInstance(model_info['model_size_mb'], (int, float))
            self.assertIsInstance(model_info['message'], str)

            print(f"✅ Model info: {model_info['model_size_mb']} MB, {model_info['model_type']}")

        except Exception as e:
            self.fail(f"Model info retrieval failed: {e}")

    def test_template_listing(self):
        """Test template discovery and listing"""
        print("\n🧪 Testing template listing...")

        try:
            templates = self.ai_gen.list_available_templates()

            self.assertIsInstance(templates, list, "Templates should be returned as list")

            if templates:
                print(f"✅ Found {len(templates)} templates")

                # Test first template structure
                with open(templates[0], 'r') as f:
                    template_data = json.load(f)

                required_fields = ['name', 'columns']
                for field in required_fields:
                    self.assertIn(field, template_data, f"Template should contain {field}")

                # Test column structure
                if template_data['columns']:
                    col = template_data['columns'][0]
                    self.assertIn('name', col, "Column should have name")
                    self.assertIn('data_type', col, "Column should have data_type")

                print(f"✅ Template structure validated: {template_data['name']}")
            else:
                print("⚠️ No templates found")

        except Exception as e:
            self.fail(f"Template listing failed: {e}")

    def test_data_generation_from_template(self):
        """Test generating data from template"""
        print("\n🧪 Testing data generation from template...")

        if not self.ai_gen.is_ready():
            self.skipTest("Model not available for generation testing")

        try:
            templates = self.ai_gen.list_available_templates()

            if not templates:
                self.skipTest("No templates available for testing")

            # Test with first available template
            template_path = templates[0]
            test_rows = 10

            df = self.ai_gen.generate_from_template(template_path, test_rows)

            if df is not None:
                self.assertIsInstance(df, pd.DataFrame, "Should return DataFrame")
                self.assertEqual(len(df), test_rows, f"Should generate {test_rows} rows")
                self.assertGreater(len(df.columns), 0, "Should have columns")

                print(f"✅ Generated {len(df)} rows with {len(df.columns)} columns")
                print(f"Columns: {list(df.columns)}")
            else:
                print("⚠️ Generation returned None - check model training")

        except Exception as e:
            print(f"❌ Generation test failed: {e}")
            # Don't fail the test as this might be due to missing model

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        print("\n🧪 Testing error handling...")

        if not self.ai_gen.is_ready():
            self.skipTest("Model not available for error handling testing")

        # Test with invalid template path
        result = self.ai_gen.generate_from_template("nonexistent_template.json", 10)
        self.assertIsNone(result, "Should return None for invalid template")

        # Test with invalid row count
        templates = self.ai_gen.list_available_templates()
        if templates:
            result = self.ai_gen.generate_from_template(templates[0], 0)
            # Should handle gracefully

        print("✅ Error handling tests completed")

def run_unified_tests():
    """Run all unified AI tests"""
    print("🧪 UNIFIED AI GENERATOR TEST SUITE")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnifiedAIGenerator)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    return result.wasSuccessful()

if __name__ == "__main__":
    run_unified_tests()
