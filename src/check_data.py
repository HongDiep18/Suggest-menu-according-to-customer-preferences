import pandas as pd
import numpy as np
import os

class DataChecker:
    def __init__(self):
        self.data_files = {
            'raw_recipes': '../data/RAW_recipes.csv',
            'raw_interactions': '../data/RAW_interactions.csv',
            'cleaned_data': '../data/cleaned_data.csv',
            'clustered_data': '../data/clustered_data.csv',
            'association_rules': '../data/association_rules.csv',
            'seasonal_trends': '../data/seasonal_trends.csv',
            'menu': '../data/menu.csv'
        }
    
    def check_file_existence(self):
        """Kiểm tra sự tồn tại của các file"""
        print("=== KIỂM TRA SỰ TỒN TẠI FILE ===")
        missing_files = []
        
        for name, path in self.data_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024  # KB
                print(f" {name}: {path} ({size:.1f} KB)")
            else:
                print(f" {name}: {path} - KHÔNG TỒN TẠI")
                missing_files.append(name)
        
        return missing_files
    
    def check_data_integrity(self):
        """Kiểm tra tính toàn vẹn dữ liệu"""
        print("\n=== KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU ===")
        
        # Kiểm tra cleaned_data
        if os.path.exists(self.data_files['cleaned_data']):
            df = pd.read_csv(self.data_files['cleaned_data'])
            print(f"Cleaned data: {len(df)} rows, {len(df.columns)} columns")
            
            # Kiểm tra missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("  Missing values found:")
                print(missing[missing > 0])
            else:
                print(" Không có missing values")
            
            # Kiểm tra rating range
            if 'rating' in df.columns:
                rating_range = f"{df['rating'].min():.1f} - {df['rating'].max():.1f}"
                print(f"Rating range: {rating_range}")
                
                if df['rating'].min() < 1 or df['rating'].max() > 5:
                    print("⚠️  Rating ngoài khoảng 1-5")
                else:
                    print(" Rating trong khoảng hợp lệ")
        
        # Kiểm tra clustered_data
        if os.path.exists(self.data_files['clustered_data']):
            df_cluster = pd.read_csv(self.data_files['clustered_data'])
            if 'cluster' in df_cluster.columns:
                n_clusters = df_cluster['cluster'].nunique()
                print(f" Số clusters: {n_clusters}")
            else:
                print(" Thiếu cột cluster")
    
    def check_data_quality(self):
        """Kiểm tra chất lượng dữ liệu"""
        print("\n=== KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ===")
        
        if os.path.exists(self.data_files['cleaned_data']):
            df = pd.read_csv(self.data_files['cleaned_data'])
            
            # Số lượng users và recipes
            n_users = df['user_id'].nunique()
            n_recipes = df['recipe_id'].nunique()
            print(f"Số users: {n_users:,}")
            print(f"Số recipes: {n_recipes:,}")
            print(f"Số interactions: {len(df):,}")
            
            # Mật độ dữ liệu
            density = len(df) / (n_users * n_recipes) * 100
            print(f"Mật độ dữ liệu: {density:.4f}%")
            
            # Phân bố rating
            if 'rating' in df.columns:
                rating_dist = df['rating'].value_counts().sort_index()
                print("\nPhân bố rating:")
                for rating, count in rating_dist.items():
                    print(f"  {rating}: {count:,} ({count/len(df)*100:.1f}%)")
    
    def check_model_outputs(self):
        """Kiểm tra kết quả các model"""
        print("\n=== KIỂM TRA KẾT QUẢ MODEL ===")
        
        # Association rules
        if os.path.exists(self.data_files['association_rules']):
            rules_df = pd.read_csv(self.data_files['association_rules'])
            print(f" Association rules: {len(rules_df)} rules")

            if len(rules_df) > 0:
                # DÙNG TÊN TIẾNG VIỆT nếu file đã đổi tên
                avg_confidence = rules_df['Xác suất Gợi ý'].mean()
                print(f"   Confidence trung bình: {avg_confidence:.3f}")
        else:
            print(" Không có association rules")

        
        # Seasonal trends
        if os.path.exists(self.data_files['seasonal_trends']):
            trends_df = pd.read_csv(self.data_files['seasonal_trends'])
            print(f" Seasonal trends: {len(trends_df)} entries")
            
            seasons = trends_df['season'].unique()
            print(f"   Các mùa: {', '.join(seasons)}")
        else:
            print(" Không có seasonal trends")
        
        # Menu
        if os.path.exists(self.data_files['menu']):
            menu_df = pd.read_csv(self.data_files['menu'])
            print(f" Menu: {len(menu_df)} items")
        else:
            print(" Không có menu")
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n" + "="*50)
        print("TỔNG HỢP KIỂM TRA DỮ LIỆU")
        print("="*50)
        
        missing_files = self.check_file_existence()
        self.check_data_integrity()
        self.check_data_quality()
        self.check_model_outputs()
        
        print("\n" + "="*50)
        if len(missing_files) == 0:
            print(" TẤT CẢ FILE ĐỀU HOÀN THÀNH!")
        else:
            print(f"  THIẾU {len(missing_files)} FILE: {', '.join(missing_files)}")
        print("="*50)
    
    def validate_recommendations(self, user_id=None):
        """Kiểm tra tính hợp lệ của recommendations"""
        print(f"\n=== KIỂM TRA RECOMMENDATION CHO USER {user_id} ===")
        
        try:
            from recommender import RestaurantRecommender
            
            recommender = RestaurantRecommender()
            if recommender.load_data(self.data_files['cleaned_data']):
                recommender.build_user_profiles()
                
                # Test với user có sẵn
                if user_id is None:
                    df = pd.read_csv(self.data_files['cleaned_data'])
                    user_id = df['user_id'].iloc[0]
                
                recs = recommender.recommend_for_user(user_id, season='Hè', n_recommendations=5)
                print(f" Tạo được {len(recs)} recommendations cho user {user_id}")
                print(f"   Recipe IDs: {recs}")
                
                return True
            else:
                print(" Không thể load dữ liệu")
                return False
                
        except Exception as e:
            print(f" Lỗi khi test recommendation: {e}")
            return False

if __name__ == "__main__":
    checker = DataChecker()
    checker.generate_summary_report()
    checker.validate_recommendations()