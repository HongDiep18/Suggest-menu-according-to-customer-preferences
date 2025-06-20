import os
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tạo thư mục lưu dữ liệu nếu chưa có
os.makedirs('../data', exist_ok=True)


class RestaurantRecommender:
    def __init__(self, max_users=10000, max_recipes=50000):
        self.data = None
        self.user_profiles = {}
        self.clusters = None
        self.association_rules_df = None
        self.seasonal_trends = None
        self.max_users = max_users
        self.max_recipes = max_recipes

    def load_data(self, data_path):
        # Tải dữ liệu và lọc top users, top recipes
        try:
            self.data = pd.read_csv(data_path)
            if self.max_users:
                top_users = self.data['user_id'].value_counts().head(self.max_users).index
                self.data = self.data[self.data['user_id'].isin(top_users)]
            if self.max_recipes:
                top_recipes = self.data['recipe_id'].value_counts().head(self.max_recipes).index
                self.data = self.data[self.data['recipe_id'].isin(top_recipes)]
            logger.info(f"Đã tải {len(self.data)} bản ghi")
            return True
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu: {e}")
            return False

    def build_user_profiles(self):
        # Xây dựng profile người dùng
        try:
            user_stats = self.data.groupby('user_id').agg({
                'rating': ['mean', 'count'],
                'minutes': 'mean',
                'calories': 'mean',
                'ingredient_count': 'mean'
            }).round(2)
            user_stats.columns = ['avg_rating', 'total_ratings', 'avg_cook_time', 'avg_calories', 'avg_ingredients']
            seasonal_prefs = self.data.groupby(['user_id', 'season'])['rating'].mean().unstack(fill_value=0)
            for user_id in user_stats.index:
                self.user_profiles[user_id] = {
                    'stats': user_stats.loc[user_id].to_dict(),
                    'seasonal_prefs': seasonal_prefs.loc[user_id].to_dict() if user_id in seasonal_prefs.index else {}
                }
            logger.info(f"Đã xây dựng profile cho {len(self.user_profiles)} người dùng")
            return self.user_profiles
        except Exception as e:
            logger.error(f"Lỗi xây dựng user profile: {e}")
            return {}

    def perform_clustering(self, n_clusters=5):
        # Phân cụm món ăn
        try:
            recipe_features = self.data.groupby('recipe_id').agg({
                'rating': 'mean',
                'minutes': 'first',
                'calories': 'first',
                'ingredient_count': 'first'
            }).dropna()
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(recipe_features)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            recipe_features['cluster'] = clusters
            recipe_features['cluster_name'] = recipe_features['cluster'].map({
                0: 'Nhanh & Nhẹ',
                1: 'Truyền thống',
                2: 'Cao cấp',
                3: 'Gia đình',
                4: 'Đặc biệt'
            })
            self.clusters = recipe_features
            self.data = pd.merge(
                self.data, recipe_features[['cluster', 'cluster_name']],
                on='recipe_id', how='left'
            )
            self.data.to_csv('../data/clustered_data.csv', index=False)
            logger.info(f"Đã phân cụm {len(recipe_features)} món ăn thành {n_clusters} nhóm")
            return recipe_features
        except Exception as e:
            logger.error(f"Lỗi phân cụm: {e}")
            return None

    def find_association_rules(self, min_support=0.005, min_confidence=0.1):
        # Tìm luật kết hợp giữa các món ăn
        try:
            user_ids = self.data['user_id'].astype('category')
            recipe_ids = self.data['recipe_id'].astype('category')
            ratings = (self.data['rating'] >= 4).astype(bool)
            user_idx = user_ids.cat.codes
            recipe_idx = recipe_ids.cat.codes
            user_item_matrix = csr_matrix(
                (ratings, (user_idx, recipe_idx)),
                shape=(len(user_ids.cat.categories), len(recipe_ids.cat.categories))
            )
            recipe_columns = [str(col) for col in recipe_ids.cat.categories]
            user_item_df = pd.DataFrame.sparse.from_spmatrix(
                user_item_matrix,
                index=user_ids.cat.categories,
                columns=recipe_columns
            )
            user_item_df = user_item_df.astype(bool)
            frequent_itemsets = apriori(user_item_df, min_support=min_support, use_colnames=True)
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                rules = rules.sort_values('confidence', ascending=False)
                rules['antecedents'] = rules['antecedents'].apply(
                    lambda x: list(x)[0] if len(x) == 1 else str(list(x))
                )
                rules['consequents'] = rules['consequents'].apply(
                    lambda x: list(x)[0] if len(x) == 1 else str(list(x))
                )
                self.association_rules_df = rules
                rules.to_csv('../data/association_rules.csv', index=False)
                logger.info(f"Tìm được {len(rules)} luật kết hợp")
                return rules
            else:
                logger.info("Không tìm được luật kết hợp nào")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Lỗi tìm luật kết hợp: {e}")
            return pd.DataFrame()

    def analyze_seasonal_trends(self):
        # Phân tích xu hướng theo mùa
        try:
            seasonal_stats = self.data.groupby('season').agg({
                'recipe_id': 'count',
                'minutes': 'mean',
                'ingredient_count': 'mean',
                'cluster': lambda x: x.mode()[0] if len(x) > 0 and pd.notna(x).any() else 0
            }).round(2).reset_index()
            seasonal_stats.columns = ['season', 'recipe_count', 'avg_minutes', 'avg_ingredients', 'popular_cluster']
            self.seasonal_trends = seasonal_stats
            seasonal_stats.to_csv('../data/seasonal_trends.csv', index=False)
            self._plot_seasonal_trends(seasonal_stats)
            logger.info("Đã phân tích xu hướng theo mùa")
            return seasonal_stats
        except Exception as e:
            logger.error(f"Lỗi phân tích xu hướng mùa: {e}")
            return None

    def _plot_seasonal_trends(self, data):
        # Vẽ biểu đồ xu hướng theo mùa
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='season', y='recipe_count', data=data)
            plt.title('Số lượng món ăn theo mùa')
            plt.ylabel('Số lượng món')
            plt.xlabel('Mùa')
            plt.tight_layout()
            plt.savefig('../data/seasonal_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Lỗi vẽ biểu đồ xu hướng mùa: {e}")

    def recommend_for_user(self, user_id, season='Hè', n_recommendations=5):
        # Gợi ý món ăn cho người dùng cụ thể
        try:
            if user_id not in self.user_profiles:
                return self._recommend_popular_items(season, n_recommendations)
            recommendations = []
            cluster_recs = self._recommend_by_cluster(user_id, max(n_recommendations // 2, 1))
            recommendations.extend(cluster_recs)
            if self.association_rules_df is not None and len(self.association_rules_df) > 0:
                rule_recs = self._recommend_by_rules(user_id, max(n_recommendations // 3, 1))
                recommendations.extend(rule_recs)
            remaining = n_recommendations - len(recommendations)
            if remaining > 0:
                seasonal_recs = self._recommend_by_season(user_id, season, remaining)
                recommendations.extend(seasonal_recs)
            unique_recs = list(dict.fromkeys(recommendations))
            if len(unique_recs) < n_recommendations:
                remaining = n_recommendations - len(unique_recs)
                popular_recs = self._recommend_popular_items(season, remaining)
                unique_recs.extend(popular_recs)
            return unique_recs[:n_recommendations]
        except Exception as e:
            logger.error(f"Lỗi gợi ý cho người dùng {user_id}: {e}")
            return []

    def _recommend_by_cluster(self, user_id, n_recs):
        if self.clusters is None:
            return []
        user_data = self.data[self.data['user_id'] == user_id]
        user_clusters = user_data.groupby('cluster')['rating'].mean().sort_values(ascending=False)
        if len(user_clusters) == 0:
            return []
        fav_cluster = user_clusters.index[0]
        cluster_recipes = self.clusters[self.clusters['cluster'] == fav_cluster].sort_values('rating', ascending=False)
        return cluster_recipes.index[:n_recs].tolist()

    def _recommend_by_rules(self, user_id, n_recs):
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return []
        user_liked = self.data[
            (self.data['user_id'] == user_id) & (self.data['rating'] >= 4)
        ]['recipe_id'].tolist()
        recommendations = []
        for recipe in user_liked:
            rules = self.association_rules_df[
                self.association_rules_df['antecedents'] == str(recipe)
            ]
            if len(rules) > 0:
                recommendations.extend([
                    int(r) if r.isdigit() else r for r in rules['consequents']
                ])
        return recommendations[:n_recs]

    def _recommend_by_season(self, user_id, season, n_recs):
        seasonal_data = self.data[self.data['season'] == season]
        popular_in_season = seasonal_data.groupby('recipe_id')['rating'].mean().sort_values(ascending=False)
        return popular_in_season.index[:n_recs].tolist()

    def _recommend_popular_items(self, season, n_recs):
        seasonal_data = self.data[self.data['season'] == season] if season else self.data
        popular_items = seasonal_data.groupby('recipe_id')['rating'].mean().sort_values(ascending=False)
        return popular_items.index[:n_recs].tolist()

    def create_menu_file(self):
        # Tạo file menu.csv phục vụ cho frontend
        try:
            menu_df = self.data.groupby(['recipe_id', 'name']).agg({
                'rating': 'mean',
                'minutes': 'first',
                'calories': 'first',
                'season': 'first',
                'ingredient_count': 'first'
            }).round(2).reset_index()
            menu_df.columns = ['id', 'name', 'avg_rating', 'minutes', 'calories', 'season', 'n_ingredients']
            menu_df['nutrition'] = menu_df['calories'].apply(lambda x: f"[{x},0,0,0,0,0,0]")
            menu_df['ingredients_list'] = menu_df['n_ingredients'].apply(
                lambda x: [f"ingredient_{i+1}" for i in range(int(x))]
            )
            menu_df['category'] = 'other'
            menu_df['price'] = 100000 + (menu_df['n_ingredients'] * 10000) + np.random.uniform(-20000, 20000, len(menu_df))
            menu_df = menu_df[[
                'id', 'name', 'minutes', 'nutrition',
                'ingredients_list', 'season', 'category', 'price'
            ]]
            menu_df.to_csv('../data/menu.csv', index=False)
            logger.info(f"Đã tạo menu với {len(menu_df)} món ăn")
            return menu_df
        except Exception as e:
            logger.error(f"Lỗi tạo file menu: {e}")
            return None


if __name__ == "__main__":
    recommender = RestaurantRecommender(max_users=10000, max_recipes=50000)
    if recommender.load_data('../data/cleaned_data.csv'):
        recommender.build_user_profiles()
        recommender.perform_clustering()
        recommender.find_association_rules()
        recommender.analyze_seasonal_trends()
        recommender.create_menu_file()
