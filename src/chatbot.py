import pandas as pd
import numpy as np
from typing import Dict, List
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FoodChatbot:
    def __init__(self, recommender_system, nlp_processor):
        """Khởi tạo chatbot với hệ thống gợi ý và NLP processor"""
        self.recommender = recommender_system
        self.nlp = nlp_processor
        self.conversation_history = []
        
        # Templates để trả lời
        self.response_templates = {
            'greeting': [
                "Xin chào! Tôi có thể giúp bạn tìm món ăn phù hợp. Bạn muốn ăn gì hôm nay?",
                "Chào bạn! Hãy cho tôi biết sở thích của bạn, tôi sẽ gợi ý món ngon nhé!",
                "Hi! Tôi là trợ lý ẩm thực. Bạn có muốn tôi gợi ý món ăn không?"
            ],
            'no_results': [
                "Xin lỗi, tôi không tìm thấy món nào phù hợp với yêu cầu của bạn.",
                "Không có món nào phù hợp. Bạn có thể thử mô tả khác không?",
                "Tôi chưa hiểu rõ yêu cầu. Bạn có thể nói cụ thể hơn được không?"
            ],
            'clarification': [
                "Bạn có thể nói rõ hơn về món ăn bạn muốn không?",
                "Tôi cần thêm thông tin. Bạn muốn món gì, vị như thế nào?",
                "Bạn có thể mô tả chi tiết hơn về sở thích của bạn không?"
            ]
        }
    
    def detect_intent_type(self, user_input: str) -> str:
        """Phát hiện loại ý định của người dùng"""
        if not isinstance(user_input, str):
            user_input = str(user_input)
        user_input = user_input.lower()
        
        # Chào hỏi
        greeting_words = ['hello', 'hi', 'chào', 'xin chào', 'hey']
        if any(word in user_input for word in greeting_words):
            return 'greeting'
        
        # Đặt món/tìm kiếm
        food_words = ['muốn', 'tìm', 'gợi ý', 'món', 'ăn', 'food', 'dish', 'want', 'recommend']
        if any(word in user_input for word in food_words):
            return 'food_request'
        
        # Hỏi thông tin
        question_words = ['gì', 'nào', 'how', 'what', 'where', 'which']
        if any(word in user_input for word in question_words):
            return 'question'
        
        return 'food_request'  # Default
    
    def generate_response(self, user_input: str) -> Dict:
        """Tạo phản hồi cho người dùng"""
        intent_type = self.detect_intent_type(user_input)
        
        response = {
            'message': '',
            'recommendations': [],
            'intent': {},
            'confidence': 0.0
        }
        
        if intent_type == 'greeting':
            response['message'] = np.random.choice(self.response_templates['greeting'])
            return response
        
        intent = self.nlp.extract_intent(user_input)
        response['intent'] = intent
        response['confidence'] = intent['confidence']
        
        logger.info(f"Intent for '{user_input}': {intent}")
        
        if intent['confidence'] < 0.05:  # Giảm ngưỡng xuống 0.05
            response['message'] = np.random.choice(self.response_templates['clarification'])
            logger.warning(f"Low confidence: {intent['confidence']} for input: {user_input}")
            return response
        
        recommendations = self.find_matching_dishes(intent, user_input)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        
        if not recommendations:
            response['message'] = np.random.choice(self.response_templates['no_results'])
            response['message'] += " Bạn có thể thử: 'Tôi muốn món Ý', 'món chay ít calo', 'món có gà'."
        else:
            response['recommendations'] = recommendations
            response['message'] = self.create_recommendation_message(intent, recommendations)
        
        self.conversation_history.append({
            'user_input': user_input,
            'intent': intent,
            'recommendations': recommendations,
            'confidence': intent['confidence']
        })
        
        return response
    
    def find_matching_dishes(self, intent: Dict, user_input: str) -> List[Dict]:
        """Tìm món ăn phù hợp với ý định"""
        try:
            if self.recommender and hasattr(self.recommender, 'data'):
                recipes_df = self.recommender.data
            else:
                data_path = '../data/cleaned_data.csv'
                if not os.path.exists(data_path):
                    logger.error("Không tìm thấy file cleaned_data.csv")
                    return []
                recipes_df = pd.read_csv(data_path)
            
            # Kiểm tra cột cần thiết
            required_columns = ['name']
            if 'tags' not in recipes_df.columns:
                recipes_df['tags'] = ''
                logger.warning("Cột 'tags' không tồn tại, tạo cột rỗng")
            if 'ingredients' not in recipes_df.columns:
                recipes_df['ingredients'] = ''
                logger.warning("Cột 'ingredients' không tồn tại, tạo cột rỗng")
            if 'calories' not in recipes_df.columns:
                recipes_df['calories'] = 0
                logger.warning("Cột 'calories' không tồn tại, tạo cột rỗng")
            if 'ingredient_count' not in recipes_df.columns:
                recipes_df['ingredient_count'] = 0
                logger.warning("Cột 'ingredient_count' không tồn tại, tạo cột rỗng")
            
            all_results = []
            
            # Semantic search
            semantic_results = self.nlp.semantic_search(user_input, recipes_df, top_k=15)
            for result in semantic_results:
                result['nutrition'] = [result.get('calories', 0)] + [0] * 6
                result['ingredient_count'] = result.get('ingredient_count', len(result.get('ingredients', '').split()))
                all_results.append(result)
            
            # Rule-based filtering
            filtered_results = self.rule_based_filter(recipes_df, intent)
            for result in filtered_results:
                result['nutrition'] = [result.get('calories', 0)] + [0] * 6
                result['ingredient_count'] = result.get('ingredient_count', len(result.get('ingredients', '').split()))
                all_results.extend(filtered_results)
            
            # Fuzzy matching
            if 'name' in recipes_df.columns:
                dish_names = recipes_df['name'].dropna().astype(str).tolist()
                fuzzy_matches = self.nlp.fuzzy_match_dishes(user_input, dish_names)
                for dish_name, score in fuzzy_matches[:5]:
                    matching_recipes = recipes_df[recipes_df['name'] == dish_name]
                    for _, recipe in matching_recipes.iterrows():
                        ingredients = recipe.get('ingredients', '')
                        ingredient_count = recipe.get('ingredient_count', len(ingredients.split()) if ingredients else 0)
                        if ingredient_count > 15:
                            logger.warning(f"Nguyên liệu bất thường cho {recipe['name']}: {ingredient_count}")
                            ingredient_count = min(ingredient_count, 10)
                        
                        all_results.append({
                            'recipe_id': recipe.get('recipe_id', recipe.get('id', '')),
                            'name': recipe.get('name', ''),
                            'score': score / 100.0,
                            'ingredients': ingredients,
                            'ingredient_count': ingredient_count,
                            'tags': recipe.get('tags', ''),
                            'nutrition': [recipe.get('calories', 0)] + [0] * 6,
                            'minutes': recipe.get('minutes', 0),
                            'method': 'fuzzy'
                        })
            
            unique_results = {}
            for result in all_results:
                recipe_id = result.get('recipe_id', result.get('name', ''))
                if recipe_id and (recipe_id not in unique_results or result['score'] > unique_results[recipe_id]['score']):
                    unique_results[recipe_id] = result
            
            final_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm món ăn: {e}")
            return []
    
    def rule_based_filter(self, recipes_df: pd.DataFrame, intent: Dict) -> List[Dict]:
        """Lọc món ăn dựa trên luật"""
        filtered_df = recipes_df.copy()
        results = []
        
        try:
            if intent['cuisine'] and 'tags' in filtered_df.columns:
                cuisine_pattern = '|'.join(intent['cuisine'])
                filtered_df = filtered_df[
                    filtered_df['tags'].str.contains(cuisine_pattern, case=False, na=False)
                ]
            
            if intent['dietary']:
                for diet in intent['dietary']:
                    if diet == 'vegetarian' and 'tags' in filtered_df.columns:
                        veg_pattern = 'vegetarian|vegan|plant.*based|veggie|chay'
                        filtered_df = filtered_df[
                            filtered_df['tags'].str.contains(veg_pattern, case=False, na=False)
                        ]
                    elif diet == 'vegan' and 'tags' in filtered_df.columns:
                        vegan_pattern = 'vegan|plant.*based|thuần chay'
                        filtered_df = filtered_df[
                            filtered_df['tags'].str.contains(vegan_pattern, case=False, na=False)
                        ]
                    elif diet == 'low_calorie':
                        if 'tags' in filtered_df.columns:
                            diet_pattern = 'low.*calorie|diet|light|healthy|ít calo'
                            filtered_df = filtered_df[
                                filtered_df['tags'].str.contains(diet_pattern, case=False, na=False)
                            ]
                        elif 'calories' in filtered_df.columns:
                            filtered_df = filtered_df[filtered_df['calories'] < 200]
                            logger.info("Sử dụng cột calories để lọc low_calorie")
            
            if intent['ingredients'] and 'ingredients' in filtered_df.columns:
                ingredient_pattern = '|'.join(intent['ingredients'])
                filtered_df = filtered_df[
                    filtered_df['ingredients'].str.contains(ingredient_pattern, case=False, na=False)
                ]
            
            for _, recipe in filtered_df.head(10).iterrows():
                ingredients = recipe.get('ingredients', '')
                ingredient_count = recipe.get('ingredient_count', len(ingredients.split()) if ingredients else 0)
                if ingredient_count > 15:
                    logger.warning(f"Nguyên liệu bất thường cho {recipe['name']}: {ingredient_count}")
                    ingredient_count = min(ingredient_count, 10)
                
                results.append({
                    'recipe_id': recipe.get('recipe_id', recipe.get('id', '')),
                    'name': recipe.get('name', ''),
                    'score': 0.8,
                    'ingredients': ingredients,
                    'ingredient_count': ingredient_count,
                    'tags': recipe.get('tags', ''),
                    'nutrition': [recipe.get('calories', 0)] + [0] * 6,
                    'minutes': recipe.get('minutes', 0),
                    'method': 'rule_based'
                })
        
        except Exception as e:
            logger.error(f"Lỗi khi lọc dựa trên luật: {e}")
        
        return results
    
    def create_recommendation_message(self, intent: Dict, recommendations: List[Dict]) -> str:
        """Tạo tin nhắn giới thiệu các món được gợi ý"""
        intro_parts = []
        
        if intent['cuisine']:
            intro_parts.append(f"món {', '.join(intent['cuisine'])}")
        
        if intent['dietary']:
            dietary_vn = {
                'vegetarian': 'chay',
                'vegan': 'thuần chay', 
                'low_calorie': 'ít calo',
                'high_protein': 'nhiều protein',
                'spicy': 'cay',
                'sweet': 'ngọt',
                'sour': 'chua',
                'salty': 'mặn'
            }
            diet_text = ', '.join([dietary_vn.get(d, d) for d in intent['dietary']])
            intro_parts.append(f"kiểu {diet_text}")
        
        if intent['ingredients']:
            intro_parts.append(f"có {', '.join(intent['ingredients'])}")
        
        # Tạo câu mở đầu
        if intro_parts:
            intro = f"Dựa trên yêu cầu {' và '.join(intro_parts)} của bạn, tôi gợi ý:"
        else:
            intro = "Tôi tìm được một số món phù hợp:"
        
        # Liệt kê các món
        dish_list = []
        for i, rec in enumerate(recommendations[:5], 1):
            name = rec.get('name', 'Món ăn')
            confidence = rec.get('score', 0) * 100
            
            # Thêm thông tin ngắn gọn
            extra_info = []
            if rec.get('tags'):
                tags = str(rec['tags'])[:50] + ('...' if len(str(rec['tags'])) > 50 else '')
                extra_info.append(tags)
            
            dish_text = f"{i}. **{name}** (độ phù hợp: {confidence:.1f}%)"
            if extra_info:
                dish_text += f" ({extra_info[0]})"
            
            dish_list.append(dish_text)
        
        message = intro + "\n\n" + "\n".join(dish_list)
        
        # Thêm lời khuyên
        if len(recommendations) > 5:
            message += f"\n\n*Còn {len(recommendations)-5} món khác nữa. Bạn có muốn xem thêm không?*"
        
        return message