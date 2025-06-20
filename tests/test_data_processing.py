import pytest
import pandas as pd
import numpy as np
import sys
import os

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor

class TestDataProcessor:
    
    def setup_method(self):
        """Thiết lập dữ liệu test"""
        self.processor = DataProcessor()
        
        # Tạo dữ liệu test recipes
        self.test_recipes = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Món 1', 'Món 2', '  Món 3  ', 'Món 4'],
            'minutes': [15, 30, 45, 999],  # 999 để test outlier
            'ingredients': [
                "['salt', 'pepper']",
                "['rice', 'chicken', 'oil']", 
                "['beef', 'onion']",
                "invalid_json"
            ],
            'nutrition': [
                "[200, 10, 5, 20, 15, 30, 8]",
                "[300, 15, 8, 25, 20, 40, 12]",
                "[400, 20, 12, 30, 25, 50, 15]",
                "invalid_nutrition"
            ]
        })
        
        # Tạo dữ liệu test interactions
        self.test_interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'recipe_id': [1, 2, 1, 3, 4],
            'rating': [5, 4, 3, 6, 2],  # 6 để test outlier
            'date': ['2023-01-15', '2023-06-20', '2023-09-10', '2023-12-05', '2023-03-25']
        })
    
    def test_extract_calories(self):
        """Test trích xuất calories"""
        processor = DataProcessor()
        
        # Test case hợp lệ
        nutrition_valid = "[200, 10, 5, 20, 15, 30, 8]"
        calories = processor._extract_calories(nutrition_valid)
        assert calories == 200
        
        # Test case không hợp lệ
        nutrition_invalid = "invalid_json"
        calories = processor._extract_calories(nutrition_invalid)
        assert calories == 0
    
    def test_count_ingredients(self):
        """Test đếm nguyên liệu"""
        processor = DataProcessor()
        
        # Test case hợp lệ
        ingredients_valid = "['salt', 'pepper', 'oil']"
        count = processor._count_ingredients(ingredients_valid)
        assert count == 3
        
        # Test case không hợp lệ
        ingredients_invalid = "invalid_json"
        count = processor._count_ingredients(ingredients_invalid)
        assert count == 0
    
    def test_get_season(self):
        """Test xác định mùa"""
        processor = DataProcessor()
        
        assert processor._get_season(1) == 'Đông'
        assert processor._get_season(4) == 'Xuân'
        assert processor._get_season(7) == 'Hè'
        assert processor._get_season(10) == 'Thu'
    
    def test_clean_recipes_data(self):
        """Test làm sạch dữ liệu recipes"""
        self.processor.recipes_df = self.test_recipes.copy()
        
        cleaned = self.processor.clean_recipes_data()
        
        # Kiểm tra đã loại bỏ outlier
        assert all(cleaned['minutes'] <= 300)
        assert all(cleaned['minutes'] > 0)
        
        # Kiểm tra đã tính calories
        assert 'calories' in cleaned.columns
        
        # Kiểm tra đã đếm ingredients
        assert 'ingredient_count' in cleaned.columns
        
        # Kiểm tra phân loại thời gian nấu
        assert 'cooking_time_category' in cleaned.columns
    
    def test_clean_interactions_data(self):
        """Test làm sạch dữ liệu interactions"""
        self.processor.interactions_df = self.test_interactions.copy()
        
        cleaned = self.processor.clean_interactions_data()
        
        # Kiểm tra rating trong khoảng 1-5
        assert all(cleaned['rating'] >= 1)
        assert all(cleaned['rating'] <= 5)
        
        # Kiểm tra đã chuyển đổi date
        assert pd.api.types.is_datetime64_any_dtype(cleaned['date'])
        
        # Kiểm tra đã thêm mùa
        assert 'season' in cleaned.columns
    
    def test_data_integrity(self):
        """Test tính toàn vẹn dữ liệu"""
        self.processor.recipes_df = self.test_recipes.copy()
        self.processor.interactions_df = self.test_interactions.copy()
        
        # Làm sạch dữ liệu
        recipes_cleaned = self.processor.clean_recipes_data()
        interactions_cleaned = self.processor.clean_interactions_data()
        
        # Kiểm tra không có giá trị null trong các cột quan trọng
        assert not recipes_cleaned['name'].isnull().any()
        assert not interactions_cleaned['rating'].isnull().any()
        
        # Kiểm tra kiểu dữ liệu
        assert pd.api.types.is_numeric_dtype(recipes_cleaned['minutes'])
        assert pd.api.types.is_numeric_dtype(interactions_cleaned['rating'])

if __name__ == "__main__":
    pytest.main([__file__])