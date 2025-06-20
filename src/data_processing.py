import pandas as pd
import numpy as np
import re
from datetime import datetime
import ast

class DataProcessor:
    def __init__(self):
        self.recipes_df = None
        self.interactions_df = None
        
    def load_raw_data(self, recipes_path, interactions_path):
        """Tải dữ liệu thô từ file CSV"""
        try:
            self.recipes_df = pd.read_csv(recipes_path)
            self.interactions_df = pd.read_csv(interactions_path)
            print(f"Đã tải {len(self.recipes_df)} công thức và {len(self.interactions_df)} tương tác")
            return True
        except Exception as e:
            print(f"Lỗi tải dữ liệu: {e}")
            return False
    
    def clean_recipes_data(self):
        """Làm sạch dữ liệu công thức"""
        if self.recipes_df is None:
            return None
            
        # Loại bỏ các hàng thiếu thông tin quan trọng
        self.recipes_df = self.recipes_df.dropna(subset=['name', 'ingredients', 'nutrition'])
        
        # Làm sạch tên món ăn
        self.recipes_df['name'] = self.recipes_df['name'].str.strip()
        
        # Xử lý thời gian nấu (minutes)
        self.recipes_df['minutes'] = pd.to_numeric(self.recipes_df['minutes'], errors='coerce')
        self.recipes_df = self.recipes_df[self.recipes_df['minutes'] > 0]
        self.recipes_df = self.recipes_df[self.recipes_df['minutes'] <= 300]  # Loại bỏ thời gian quá dài
        
        # Xử lý nutrition (lấy calories)
        self.recipes_df['calories'] = self.recipes_df['nutrition'].apply(self._extract_calories)
        
        # Xử lý ingredients
        self.recipes_df['ingredient_count'] = self.recipes_df['ingredients'].apply(self._count_ingredients)
        
        # Phân loại theo thời gian nấu
        self.recipes_df['cooking_time_category'] = pd.cut(
            self.recipes_df['minutes'], 
            bins=[0, 15, 30, 60, 300], 
            labels=['Nhanh', 'Trung bình', 'Lâu', 'Rất lâu']
        )
        
        return self.recipes_df
    
    def _extract_calories(self, nutrition_str):
        """Trích xuất calories từ chuỗi nutrition"""
        try:
            nutrition_list = ast.literal_eval(nutrition_str)
            return nutrition_list[0] if len(nutrition_list) > 0 else 0
        except:
            return 0
    
    def _count_ingredients(self, ingredients_str):
        """Đếm số lượng nguyên liệu"""
        try:
            ingredients_list = ast.literal_eval(ingredients_str)
            return len(ingredients_list)
        except:
            return 0
    
    def clean_interactions_data(self):
        """Làm sạch dữ liệu tương tác"""
        if self.interactions_df is None:
            return None
            
        # Chỉ giữ rating từ 1-5
        self.interactions_df = self.interactions_df[
            (self.interactions_df['rating'] >= 1) & 
            (self.interactions_df['rating'] <= 5)
        ]
        
        # Chuyển đổi date
        self.interactions_df['date'] = pd.to_datetime(self.interactions_df['date'])
        
        # Thêm thông tin mùa
        self.interactions_df['season'] = self.interactions_df['date'].dt.month.apply(self._get_season)
        
        return self.interactions_df
    
    def _get_season(self, month):
        """Xác định mùa từ tháng"""
        if month in [12, 1, 2]:
            return 'Đông'
        elif month in [3, 4, 5]:
            return 'Xuân'
        elif month in [6, 7, 8]:
            return 'Hè'
        else:
            return 'Thu'
    
    def merge_and_save(self, output_path):
        """Kết hợp dữ liệu và lưu file"""
        if self.recipes_df is None or self.interactions_df is None:
            return False
            
        # Kết hợp dữ liệu
        merged_df = pd.merge(
            self.interactions_df, 
            self.recipes_df, 
            left_on='recipe_id', 
            right_on='id', 
            how='inner'
        )
        
        # Chọn các cột cần thiết
        columns_to_keep = [
            'user_id', 'recipe_id', 'rating', 'date', 'season',
            'name', 'minutes', 'calories', 'ingredient_count', 'cooking_time_category'
        ]
        
        cleaned_df = merged_df[columns_to_keep]
        
        # Lưu file
        cleaned_df.to_csv(output_path, index=False)
        print(f"Đã lưu {len(cleaned_df)} bản ghi vào {output_path}")
        
        return cleaned_df

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Xử lý dữ liệu
    if processor.load_raw_data('../data/RAW_recipes.csv', '../data/RAW_interactions.csv'):
        processor.clean_recipes_data()
        processor.clean_interactions_data()
        processor.merge_and_save('../data/cleaned_data.csv')