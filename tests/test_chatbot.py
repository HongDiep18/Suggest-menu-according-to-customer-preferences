# tests/test_chatbot.py
import unittest
import sys
import os
import pandas as pd

# Thêm thư mục src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_processor import NLPProcessor
from chatbot import FoodChatbot

class TestNLPProcessor(unittest.TestCase):
    """Test class NLPProcessor"""
    
    def setUp(self):
        """Khởi tạo NLP processor cho test"""
        self.nlp = NLPProcessor()
    
    def test_intent_extraction_vegetarian(self):
        """Test trích xuất ý định món chay"""
        intent = self.nlp.extract_intent("Tôi muốn món chay ít calo")
        
        self.assertIn('vegetarian', intent['dietary'])
        self.assertIn('low_calorie', intent['dietary'])
        self.assertGreater(intent['confidence'], 0)
        print(f"✅ Vegetarian intent: {intent}")
    
    def test_intent_extraction_cuisine(self):
        """Test trích xuất ý định về ẩm thực"""
        test_cases = [
            ("Gợi ý món Ý có pasta", ['italian'], ['noodles']),
            ("Tôi muốn ăn phở Việt Nam", ['vietnamese'], []),
            ("Món Nhật có sushi", ['japanese'], []),
        ]
        
        for text, expected_cuisine, expected_ingredients in test_cases:
            with self.subTest(text=text):
                intent = self.nlp.extract_intent(text)
                
                for cuisine in expected_cuisine:
                    self.assertIn(cuisine, intent['cuisine'])
                
                for ingredient in expected_ingredients:
                    self.assertIn(ingredient, intent['ingredients'])
                
                print(f"✅ Cuisine test '{text}': {intent}")
    
    def test_intent_extraction_ingredients(self):
        """Test trích xuất nguyên liệu"""
        test_cases = [
            ("Món có gà và rau", ['chicken', 'vegetables']),
            ("Tôi muốn ăn beef steak", ['beef']),
            ("Làm món với tôm", ['shrimp']),
        ]
        
        for text, expected_ingredients in test_cases:
            with self.subTest(text=text):
                intent = self.nlp.extract_intent(text)
                
                for ingredient in expected_ingredients:
                    self.assertIn(ingredient, intent['ingredients'])
                
                print(f"✅ Ingredients test '{text}': {intent}")
    
    def test_intent_extraction_dietary(self):
        """Test trích xuất chế độ ăn"""
        test_cases = [
            ("Món chay healthy", ['vegetarian', 'low_calorie']),
            ("Tôi ăn kiêng", ['low_calorie']),
            ("Món cay nha", ['spicy']),
            ("Dessert ngọt", ['sweet']),
        ]
        
        for text, expected_dietary in test_cases:
            with self.subTest(text=text):
                intent = self.nlp.extract_intent(text)
                
                for diet in expected_dietary:
                    self.assertIn(diet, intent['dietary'])
                
                print(f"✅ Dietary test '{text}': {intent}")
    
    def test_fuzzy_matching(self):
        """Test tìm kiếm mờ tên món ăn"""
        dish_names = [
            "Chicken Parmesan",
            "Beef Stroganoff", 
            "Vegetable Stir Fry",
            "Salmon Teriyaki",
            "Chocolate Cake"
        ]
        
        test_cases = [
            ("chicken parm", "Chicken Parmesan"),
            ("beef strog", "Beef Stroganoff"),
            ("veggie stir", "Vegetable Stir Fry"),
        ]
        
        for query, expected in test_cases:
            with self.subTest(query=query):
                matches = self.nlp.fuzzy_match_dishes(query, dish_names, threshold=60)
                
                self.assertTrue(len(matches) > 0)
                # Kiểm tra kết quả đầu tiên có phải là expected không
                best_match = matches[0][0]
                self.assertEqual(best_match, expected)
                
                print(f"✅ Fuzzy match '{query}' -> '{best_match}' (score: {matches[0][1]})")
    
    def test_semantic_search(self):
        """Test tìm kiếm ngữ nghĩa (cần có dữ liệu)"""
        # Tạo dữ liệu giả để test
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': [
                'Grilled Chicken Salad',
                'Vegetarian Pizza', 
                'Beef Burger',
                'Fish and Chips',
                'Chocolate Ice Cream'
            ],
            'ingredients': [
                'chicken breast, lettuce, tomato, cucumber',
                'tomato sauce, cheese, bell pepper, mushroom',
                'ground beef, bun, lettuce, tomato',
                'fish fillet, potato, oil',
                'milk, cream, chocolate, sugar'
            ],
            'tags': [
                'healthy, protein, salad',
                'vegetarian, pizza, cheese',
                'meat, burger, fast food',
                'seafood, fried, british',
                'dessert, sweet, cold'
            ]
        })
        
        test_queries = [
            "healthy chicken dish",
            "vegetarian food",
            "sweet dessert"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                results = self.nlp.semantic_search(query, test_data, top_k=3)
                
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0)
                
                # Kiểm tra kết quả có score > 0
                for result in results:
                    self.assertGreater(result['score'], 0)
                    self.assertIn('name', result)
                
                print(f"✅ Semantic search '{query}': {len(results)} results")
                for r in results[:2]:
                    print(f"   - {r['name']} (score: {r['score']:.3f})")


class TestFoodChatbot(unittest.TestCase):
    """Test class FoodChatbot"""
    
    def setUp(self):
        """Khởi tạo chatbot cho test"""
        self.nlp = NLPProcessor()
        # Mock recommender system (không cần thật)
        self.chatbot = FoodChatbot(None, self.nlp)
    
    def test_intent_detection(self):
        """Test phát hiện loại ý định"""
        test_cases = [
            ("Xin chào", "greeting"),
            ("Hi bot", "greeting"),
            ("Tôi muốn món gì đó", "food_request"),
            ("Gợi ý món ăn", "food_request"),
            ("Có món gì ngon?", "question"),
        ]
        
        for text, expected_intent in test_cases:
            with self.subTest(text=text):
                detected_intent = self.chatbot.detect_intent_type(text)
                self.assertEqual(detected_intent, expected_intent)
                print(f"✅ Intent detection '{text}' -> '{detected_intent}'")
    
    def test_response_generation(self):
        """Test tạo phản hồi"""
        test_inputs = [
            "Xin chào",
            "Tôi muốn món chay",
            "Gợi ý món Ý",
            "Có gì ngon không?"
        ]
        
        for user_input in test_inputs:
            with self.subTest(input=user_input):
                response = self.chatbot.generate_response(user_input)
                
                # Kiểm tra response có đúng format không
                self.assertIsInstance(response, dict)
                self.assertIn('message', response)
                self.assertIn('recommendations', response)
                self.assertIn('intent', response)
                self.assertIn('confidence', response)
                
                # Message không được rỗng
                self.assertTrue(len(response['message']) > 0)
                
                print(f"✅ Response for '{user_input}':")
                print(f"   Message: {response['message'][:100]}...")
                print(f"   Recommendations: {len(response['recommendations'])}")
                print(f"   Confidence: {response['confidence']:.3f}")
    
    def test_conversation_history(self):
        """Test lưu lịch sử hội thoại"""
        initial_history_length = len(self.chatbot.conversation_history)
        
        # Gửi một số tin nhắn
        test_messages = [
            "Tôi muốn món chay",
            "Có món Ý không?",
            "Gợi ý món ngọt"
        ]
        
        for message in test_messages:
            self.chatbot.generate_response(message)
        
        # Kiểm tra lịch sử có được lưu không
        final_history_length = len(self.chatbot.conversation_history)
        self.assertEqual(final_history_length, initial_history_length + len(test_messages))
        
        print(f"✅ Conversation history: {final_history_length} messages saved")
    
    def test_rule_based_filter(self):
        """Test lọc dựa trên luật"""
        # Tạo dữ liệu giả
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Veggie Pizza', 'Chicken Curry', 'Beef Steak', 'Fish Salad'],
            'ingredients': ['cheese, vegetables', 'chicken, spices', 'beef, potato', 'fish, lettuce'],
            'tags': ['vegetarian, italian', 'spicy, indian', 'meat, grilled', 'healthy, seafood']
        })
        
        # Test các intent khác nhau
        test_intents = [
            {
                'cuisine': ['italian'],
                'dietary': [],
                'ingredients': [],
                'meal_time': []
            },
            {
                'cuisine': [],
                'dietary': ['vegetarian'],
                'ingredients': [],
                'meal_time': []
            },
            {
                'cuisine': [],
                'dietary': [],
                'ingredients': ['chicken'],
                'meal_time': []
            }
        ]
        
        for i, intent in enumerate(test_intents):
            with self.subTest(intent=i):
                results = self.chatbot.rule_based_filter(test_data, intent)
                
                self.assertIsInstance(results, list)
                print(f"✅ Rule-based filter test {i+1}: {len(results)} results")
                
                for result in results:
                    self.assertIn('name', result)
                    self.assertIn('score', result)


def run_specific_tests():
    """Chạy một số test cụ thể cho demo"""
    print("🧪 CHẠY TESTS CHO CHATBOT")
    print("=" * 50)
    
    # Test NLP Processor
    print("\n1. TEST NLP PROCESSOR")
    nlp = NLPProcessor()
    
    # Test trích xuất intent
    test_sentences = [
        "Tôi muốn món chay ít calo",
        "Gợi ý món Ý có gà",
        "Món Việt Nam cay cay",
        "Dessert ngọt cho bữa tối"
    ]
    
    for sentence in test_sentences:
        intent = nlp.extract_intent(sentence)
        print(f"   '{sentence}'")
        print(f"   -> Cuisine: {intent['cuisine']}")
        print(f"   -> Dietary: {intent['dietary']}")
        print(f"   -> Ingredients: {intent['ingredients']}")
        print(f"   -> Confidence: {intent['confidence']:.3f}")
        print()
    
    # Test Chatbot
    print("\n2. TEST CHATBOT")
    chatbot = FoodChatbot(None, nlp)
    
    test_messages = [
        "Xin chào",
        "Tôi muốn món chay healthy",
        "Có món Ý nào ngon không?"
    ]
    
    for message in test_messages:
        response = chatbot.generate_response(message)
        print(f"   User: '{message}'")
        print(f"   Bot: {response['message'][:100]}...")
        print(f"   Recommendations: {len(response['recommendations'])}")
        print()
    
    print("✅ TẤT CẢ TESTS ĐÃ HOÀN THÀNH!")


if __name__ == '__main__':
    # Chạy specific tests trước
    run_specific_tests()
    
    print("\n" + "="*50)
    print("CHẠY UNIT TESTS CHI TIẾT:")
    print("="*50)
    
    # Chạy unit tests
    unittest.main(verbosity=2)