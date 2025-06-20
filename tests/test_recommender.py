import pytest
import pandas as pd
import numpy as np
import sys
import os

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommender import RestaurantRecommender

class TestRestaurantRecommender:
    
    def setup_method(self):
        """Thiết lập dữ liệu test"""
        self.recommender = RestaurantRecommender()
        
        # Tạo dữ liệu test
        self.test_data = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
            'recipe_id': [1, 2, 3, 1, 4, 2, 3, 4],
            'rating': [5, 4, 3, 4, 5, 3, 4, 2],
            'date': pd.date_range('2023-01-01', periods=8, freq='M'),
            'season': ['Đông', 'Đông', 'Xuân', 'Xuân', 'Hè', 'Hè', 'Thu', 'Thu'],
            'name': ['Món 1', 'Món 2', 'Món 3', 'Món 1', 'Món 4', 'Món 2', 'Món 3', 'Món 4'],
            'minutes': [15, 30, 45, 15, 20, 30, 45, 25],
            'calories': [200, 300, 400, 200, 250, 300, 400, 280],
            'ingredient_count': [3, 5, 7, 3, 4, 5, 7, 6],
            'cooking_time_category': ['Nhanh', 'Trung bình', 'Lâu', 'Nhanh', 'Nhanh', 'Trung bình', 'Lâu', 'Trung bình']
        })
        
        self.recommender.data = self.test_data
    
    def test_build_user_profiles(self):
        """Test xây dựng user profiles"""
        profiles = self.recommender.build_user_profiles()
        
        # Kiểm tra có tạo profile cho tất cả users
        assert len(profiles) == 3  # 3 users unique
        
        # Kiểm tra structure của profile
        user_1_profile = profiles[1]
        assert 'stats' in user_1_profile
        assert 'seasonal_prefs' in user_1_profile
        
        # Kiểm tra stats
        stats = user_1_profile['stats']
        assert 'avg_rating' in stats
        assert 'total_ratings' in stats
        assert 'avg_cook_time' in stats
    
    def test_clustering(self):
        """Test phân cụm"""
        clusters = self.recommender.perform_clustering(n_clusters=2)
        
        # Kiểm tra có tạo clusters
        assert clusters is not None
        assert len(clusters) > 0
        
        # Kiểm tra có cột cluster
        assert 'cluster' in clusters.columns
        assert 'cluster_name' in clusters.columns
        
        # Kiểm tra số clusters
        n_unique_clusters = clusters['cluster'].nunique()
        assert n_unique_clusters <= 2
    
    def test_seasonal_trends(self):
        """Test phân tích xu hướng mùa"""
        trends = self.recommender.analyze_seasonal_trends()
        
        # Kiểm tra có dữ liệu trends
        assert trends is not None
        assert len(trends) > 0
        
        # Kiểm tra các cột cần thiết
        required_columns = ['season', 'cooking_time_category', 'avg_rating']
        for col in required_columns:
            assert col in trends.columns
    
    def test_recommend_for_user(self):
        """Test gợi ý cho user"""
        # Xây dựng user profiles trước
        self.recommender.build_user_profiles()
        
        # Test gợi ý cho user có sẵn
        recommendations = self.recommender.recommend_for_user(user_id=1, season='Hè', n_recommendations=3)
        
        # Kiểm tra có trả về recommendations
        assert recommendations is not None
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
    
    def test_recommend_for_new_user(self):
        """Test gợi ý cho user mới"""
        # Xây dựng user profiles trước
        self.recommender.build_user_profiles()
        
        # Test với user không tồn tại
        recommendations = self.recommender.recommend_for_user(user_id=999, season='Hè', n_recommendations=3)
        
        # Vẫn phải trả về recommendations (popular items)
        assert recommendations is not None
        assert isinstance(recommendations, list)
    
    def test_recommend_by_cluster(self):
        """Test gợi ý theo cluster"""
        # Setup clustering
        self.recommender.perform_clustering(n_clusters=2)
        self.recommender.build_user_profiles()
        
        # Test gợi ý
        cluster_recs = self.recommender._recommend_by_cluster(user_id=1, n_recs=2)
        
        assert isinstance(cluster_recs, list)
        assert len(cluster_recs) <= 2
    
    def test_recommend_by_season(self):
        """Test gợi ý theo mùa"""
        seasonal_recs = self.recommender._recommend_by_season(user_id=1, season='Hè', n_recs=2)
        
        assert isinstance(seasonal_recs, list)
        assert len(seasonal_recs) <= 2
    
    def test_popular_items_fallback(self):
        """Test fallback với popular items"""
        popular_recs = self.recommender._recommend_popular_items(season='Hè', n_recs=3)
        
        assert isinstance(popular_recs, list)
        assert len(popular_recs) <= 3
    
    def test_data_validation(self):
        """Test validation dữ liệu"""
        # Test với dữ liệu rỗng
        empty_recommender = RestaurantRecommender()
        empty_recommender.data = pd.DataFrame()
        
        profiles = empty_recommender.build_user_profiles()
        assert len(profiles) == 0
        
        # Test clustering với dữ liệu không đủ
        clusters = empty_recommender.perform_clustering()
        assert clusters is None or len(clusters) == 0

if __name__ == "__main__":
    pytest.main([__file__])