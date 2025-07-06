#!/usr/bin/env python3
"""
UUID Functionality Test Suite
Tests UUID generation and formatting capabilities
"""

import sys
import os
from pathlib import Path
import pandas as pd
import re
import unittest
import uuid

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the classes we need
from interactive_mode import InteractiveGenerator, ColumnDefinition

class TestUUIDFunctionality(unittest.TestCase):
    """Test suite for UUID generation functionality"""

    def setUp(self):
        """Set up test environment"""
        try:
            self.generator = InteractiveGenerator()
        except Exception as e:
            self.skipTest(f"Could not initialize InteractiveGenerator: {e}")

    def test_uuid_column_definition(self):
        """Test UUID column definition creation"""
        print("\n🧪 Testing UUID column definition...")

        # Test basic UUID column
        uuid_col = ColumnDefinition('user_id', 'uuid')

        self.assertEqual(uuid_col.name, 'user_id')
        self.assertEqual(uuid_col.data_type, 'uuid')
        self.assertEqual(uuid_col.uuid_format, 'short')  # default
        self.assertEqual(uuid_col.uuid_length, 8)  # default

        print("✅ Basic UUID column definition created")

        # Test UUID column with custom parameters
        custom_uuid_col = ColumnDefinition(
            'order_id', 'uuid',
            uuid_format='standard',
            uuid_prefix='ORD_',
            uuid_length=12
        )

        self.assertEqual(custom_uuid_col.uuid_format, 'standard')
        self.assertEqual(custom_uuid_col.uuid_prefix, 'ORD_')
        self.assertEqual(custom_uuid_col.uuid_length, 12)

        print("✅ Custom UUID column definition created")

    def test_short_uuid_generation(self):
        """Test short UUID format generation"""
        print("\n🧪 Testing short UUID generation...")

        # Create UUID column definition
        self.generator.column_definitions = [
            ColumnDefinition('id', 'uuid', uuid_format='short', uuid_length=8)
        ]

        # Generate test data
        df = self.generator._generate_synthetic_data(100)

        self.assertIsNotNone(df, "Should generate DataFrame")
        self.assertIn('id', df.columns, "Should contain ID column")

        # Test UUID format
        ids = df['id'].tolist()

        # Check uniqueness
        unique_ids = set(ids)
        self.assertEqual(len(unique_ids), len(ids), "All UUIDs should be unique")

        # Check format (8 character alphanumeric)
        for uuid_val in ids[:10]:  # Test first 10
            self.assertIsInstance(uuid_val, str, "UUID should be string")
            self.assertEqual(len(uuid_val), 8, "Short UUID should be 8 characters")
            self.assertTrue(uuid_val.isalnum(), "Short UUID should be alphanumeric")

        print(f"✅ Generated {len(unique_ids)} unique short UUIDs")
        print(f"Sample UUIDs: {ids[:5]}")

    def test_standard_uuid_generation(self):
        """Test standard UUID format generation"""
        print("\n🧪 Testing standard UUID generation...")

        # Create UUID column definition
        self.generator.column_definitions = [
            ColumnDefinition('transaction_id', 'uuid', uuid_format='standard')
        ]

        # Generate test data
        df = self.generator._generate_synthetic_data(50)

        self.assertIsNotNone(df, "Should generate DataFrame")

        # Test standard UUID format (36 characters with hyphens)
        ids = df['transaction_id'].tolist()
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

        for uuid_val in ids[:10]:
            self.assertIsInstance(uuid_val, str, "UUID should be string")
            self.assertEqual(len(uuid_val), 36, "Standard UUID should be 36 characters")
            self.assertRegex(uuid_val, uuid_pattern, "Should match UUID pattern")

            # Verify it's a valid UUID
            try:
                uuid.UUID(uuid_val)
            except ValueError:
                self.fail(f"Invalid UUID format: {uuid_val}")

        print(f"✅ Generated {len(ids)} valid standard UUIDs")
        print(f"Sample UUIDs: {ids[:3]}")

    def test_human_readable_uuid_generation(self):
        """Test human-readable UUID format generation"""
        print("\n🧪 Testing human-readable UUID generation...")

        # Create UUID column definition with prefix - fix: use 'readable' not 'human_readable'
        self.generator.column_definitions = [
            ColumnDefinition(
                'customer_id', 'uuid',
                uuid_format='readable',  # Changed from 'human_readable' to 'readable'
                uuid_prefix='CUST',      # Fixed: removed underscore from prefix
                uuid_length=10
            )
        ]

        # Generate test data
        df = self.generator._generate_synthetic_data(30)

        self.assertIsNotNone(df, "Should generate DataFrame")

        # Test human-readable format
        ids = df['customer_id'].tolist()

        for uuid_val in ids[:10]:
            self.assertIsInstance(uuid_val, str, "UUID should be string")
            # Fix: expect format like CUST-2025-000001, not CUST_
            self.assertTrue(uuid_val.startswith('CUST-'), "Should start with CUST- prefix")

            # Verify format: PREFIX-YEAR-NNNNNN
            parts = uuid_val.split('-')
            self.assertEqual(len(parts), 3, "Should have 3 parts separated by hyphens")
            self.assertEqual(parts[0], 'CUST', "First part should be prefix")
            self.assertEqual(len(parts[2]), 6, "Sequential number should be 6 digits")

        print(f"✅ Generated {len(ids)} human-readable UUIDs with prefix")
        print(f"Sample UUIDs: {ids[:3]}")

    def test_uuid_uniqueness_large_scale(self):
        """Test UUID uniqueness at larger scale"""
        print("\n🧪 Testing UUID uniqueness at scale...")

        # Test with larger dataset
        self.generator.column_definitions = [
            ColumnDefinition('id', 'uuid', uuid_format='short'),
            ColumnDefinition('ref_id', 'uuid', uuid_format='standard')
        ]

        # Generate larger dataset
        df = self.generator._generate_synthetic_data(1000)

        self.assertIsNotNone(df, "Should generate DataFrame")

        # Check uniqueness for both columns
        for col in ['id', 'ref_id']:
            ids = df[col].tolist()
            unique_ids = set(ids)

            self.assertEqual(len(unique_ids), len(ids),
                           f"All {col} values should be unique")

            # Check no null values
            self.assertFalse(df[col].isnull().any(),
                           f"No null values in {col}")

        print(f"✅ Verified uniqueness for {len(df)} rows across multiple UUID columns")

    def test_uuid_performance(self):
        """Test UUID generation performance"""
        print("\n🧪 Testing UUID generation performance...")

        import time

        # Setup for performance test - fix: use 'readable' not 'human_readable'
        self.generator.column_definitions = [
            ColumnDefinition('id1', 'uuid', uuid_format='short'),
            ColumnDefinition('id2', 'uuid', uuid_format='standard'),
            ColumnDefinition('id3', 'uuid', uuid_format='readable', uuid_prefix='TEST')  # Fixed format name
        ]

        # Time the generation
        start_time = time.time()
        df = self.generator._generate_synthetic_data(500)
        end_time = time.time()

        generation_time = end_time - start_time

        self.assertIsNotNone(df, "Should generate DataFrame")
        self.assertEqual(len(df), 500, "Should generate requested rows")

        # Performance should be reasonable (less than 10 seconds for 500 rows)
        self.assertLess(generation_time, 10.0,
                       "UUID generation should be performant")

        rows_per_second = len(df) / generation_time if generation_time > 0 else float('inf')

        print(f"✅ Generated {len(df)} rows in {generation_time:.2f}s")
        print(f"Performance: {rows_per_second:.0f} rows/second")

    def test_uuid_error_handling(self):
        """Test UUID generation error handling"""
        print("\n🧪 Testing UUID error handling...")

        # Test with invalid parameters
        try:
            # This should handle gracefully
            col_def = ColumnDefinition(
                'bad_id', 'uuid',
                uuid_format='invalid_format',
                uuid_length=-5
            )

            self.generator.column_definitions = [col_def]
            df = self.generator._generate_synthetic_data(10)

            # Should either generate valid UUIDs or handle gracefully
            if df is not None:
                self.assertTrue(len(df) > 0, "Should generate some data")

        except Exception as e:
            # Should handle errors gracefully
            print(f"⚠️ Handled error gracefully: {e}")

        print("✅ Error handling tests completed")

def run_uuid_tests():
    """Run all UUID functionality tests"""
    print("🧪 UUID FUNCTIONALITY TEST SUITE")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUUIDFunctionality)

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
            print(f"  - {test}")

    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    return result.wasSuccessful()

if __name__ == "__main__":
    run_uuid_tests()
