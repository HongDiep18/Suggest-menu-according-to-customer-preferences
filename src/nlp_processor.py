import nltk
import spacy
import re
import pandas as pd
from typing import List, Dict, Tuple, Set
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from unidecode import unidecode

class NLPProcessor:
    def __init__(self):
        """Khởi tạo bộ xử lý NLP với hỗ trợ tiếng Việt mạnh"""
        self.setup_nlp()
        self.load_food_keywords()
        self.setup_vietnamese_stopwords()
    
    def setup_nlp(self):
        """Cài đặt các thư viện NLP"""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Không thể tải spaCy model, sử dụng chế độ cơ bản")
                self.nlp = None
        except Exception as e:
            print(f"Lỗi khi cài đặt NLP: {e}")
            self.nlp = None
    
    def setup_vietnamese_stopwords(self):
        """Thiết lập từ dừng tiếng Việt"""
        self.vietnamese_stopwords = {
            'tôi', 'toi', 'mình', 'minh', 'em', 'anh', 'chị', 'chi', 'của', 'cua',
            'và', 'va', 'với', 'voi', 'cho', 'để', 'de', 'từ', 'tu', 'trong', 'ngoài',
            'ngoai', 'trên', 'tren', 'dưới', 'duoi', 'này', 'nay', 'đó', 'do',
            'rất', 'rat', 'lắm', 'lam', 'nhiều', 'nhieu', 'ít', 'it', 'một', 'mot',
            'có', 'co', 'không', 'khong', 'được', 'duoc', 'là', 'la', 'sẽ', 'se',
            'đã', 'da', 'đang', 'dang', 'bị', 'bi', 'được', 'duoc', 'hay', 'hoặc', 'hoac'
        }
    
    def load_food_keywords(self):
        """Tải từ khóa thực phẩm và thuộc tính với hỗ trợ tiếng Việt mạnh"""
        
        # Ẩm thực theo vùng miền
        self.cuisine_keywords = {
            'vietnamese': [
                'vietnamese', 'vietnam', 'việt nam', 'viet nam', 'việt', 'viet',
                'pho', 'phở', 'bun', 'bún', 'banh', 'bánh', 'che', 'chè', 'goi', 'gỏi',
                'nem', 'spring roll', 'chả cá', 'cha ca', 'bún bò', 'bun bo',
                'cơm tấm', 'com tam', 'bánh mì', 'banh mi', 'bánh xèo', 'banh xeo',
                'cao lầu', 'cao lau', 'mì quảng', 'mi quang', 'hủ tiếu', 'hu tieu',
                'bún riêu', 'bun rieu', 'bánh cuốn', 'banh cuon', 'chả cá lã vọng',
                'bún chả', 'bun cha', 'phở gà', 'pho ga', 'phở bò', 'pho bo',
                'cháo', 'chao', 'xôi', 'xoi', 'bánh chưng', 'banh chung'
            ],
            'chinese': [
                'chinese', 'china', 'trung quốc', 'trung quoc', 'tàu', 'tau',
                'dim sum', 'dumpling', 'há cảo', 'ha cao', 'sủi cảo', 'sui cao',
                'mì xào', 'mi xao', 'cơm chiên', 'com chien', 'thịt xá xíu', 'thit xa xiu',
                'kung pao', 'mapo tofu', 'peking duck', 'vịt quay', 'vit quay',
                'lẩu tứ xuyên', 'lau tu xuyen', 'món tàu', 'mon tau', 'dimsum'
            ],
            'italian': [
                'italian', 'italy', 'ý', 'italia', 'pasta', 'pizza', 'spaghetti',
                'lasagna', 'carbonara', 'bolognese', 'risotto', 'ravioli',
                'gnocchi', 'tiramisu', 'gelato', 'bruschetta', 'focaccia',
                'mì ý', 'mi y', 'bánh pizza', 'banh pizza'
            ],
            'mexican': [
                'mexican', 'mexico', 'taco', 'burrito', 'quesadilla', 'salsa',
                'guacamole', 'nachos', 'enchilada', 'fajita', 'churro',
                'mexico', 'mex', 'món mê-hi-cô', 'mon me-hi-co'
            ],
            'indian': [
                'indian', 'india', 'ấn độ', 'an do', 'curry', 'cà ri', 'ca ri',
                'biryani', 'tandoori', 'masala', 'naan', 'samosa', 'dal',
                'tikka', 'vindaloo', 'korma', 'rogan josh', 'lassi'
            ],
            'japanese': [
                'japanese', 'japan', 'nhật bản', 'nhat ban', 'nhật', 'nhat',
                'sushi', 'sashimi', 'ramen', 'tempura', 'miso', 'udon', 'soba',
                'yakitori', 'tonkatsu', 'okonomiyaki', 'takoyaki', 'bento',
                'mochi', 'wasabi', 'teriyaki', 'katsu', 'gyoza'
            ],
            'thai': [
                'thai', 'thailand', 'thái lan', 'thai lan', 'thái', 'thai',
                'tom yum', 'pad thai', 'green curry', 'som tam', 'massaman',
                'pad see ew', 'larb', 'sticky rice', 'mango sticky rice',
                'tom kha', 'red curry', 'panang', 'thai basil'
            ],
            'korean': [
                'korean', 'korea', 'hàn quốc', 'han quoc', 'hàn', 'han',
                'kimchi', 'bulgogi', 'bibimbap', 'korean bbq', 'galbi',
                'japchae', 'tteokbokki', 'samgyeopsal', 'hotteok', 'banchan'
            ]
        }
        
        # Sở thích ăn uống và chế độ ăn
        self.dietary_keywords = {
            'vegetarian': [
                'vegetarian', 'veggie', 'plant based', 'no meat', 'chay', 'ăn chay', 'an chay',
                'không thịt', 'khong thit', 'thuần chay', 'thuan chay', 'chay trường',
                'chay truong', 'không động vật', 'khong dong vat', 'rau củ', 'rau cu'
            ],
            'vegan': [
                'vegan', 'plant only', 'dairy free', 'no animal', 'thuần chay', 'thuan chay',
                'hoàn toàn chay', 'hoan toan chay', 'không sữa', 'khong sua',
                'không trứng', 'khong trung', 'plant-based'
            ],
            'low_calorie': [
                'low calorie', 'diet', 'light', 'healthy', 'low cal', 'ít calo', 'it calo',
                'giảm cân', 'giam can', 'ăn kiêng', 'an kieng', 'healthy', 'lành mạnh',
                'lanh manh', 'ít béo', 'it beo', 'low fat', 'fitness'
            ],
            'high_protein': [
                'high protein', 'protein rich', 'muscle', 'gym', 'nhiều protein', 'nhieu protein',
                'đạm cao', 'dam cao', 'tăng cơ', 'tang co', 'bodybuilding',
                'protein tinggi', 'nhiều đạm', 'nhieu dam'
            ],
            'gluten_free': [
                'gluten free', 'no gluten', 'không gluten', 'khong gluten',
                'không chứa gluten', 'khong chua gluten', 'celiac', 'wheat free'
            ],
            'diabetic': [
                'diabetic', 'diabetes', 'tiểu đường', 'tieu duong', 'đái tháo đường',
                'dai thao duong', 'ít đường', 'it duong', 'no sugar', 'sugar free'
            ]
        }
        
        # Vị giác và độ cay
        self.taste_keywords = {
            'spicy': [
                'spicy', 'hot', 'chili', 'pepper', 'fire', 'cay', 'cấy', 'cap',
                'ớt', 'ot', '매운', 'cay nồng', 'cay nong', 'cực cay', 'cuc cay',
                'siêu cay', 'sieu cay', 'tê lưỡi', 'te luoi', 'cay như lửa',
                'cay nhu lua', 'cháy miệng', 'chay mieng'
            ],
            'sweet': [
                'sweet', 'dessert', 'sugar', 'cake', 'candy', 'ngọt', 'ngot',
                'đường', 'duong', 'bánh ngọt', 'banh ngot', 'tráng miệng', 'trang mieng',
                'chè', 'che', 'kem', 'bánh', 'banh', 'kẹo', 'keo', 'mật', 'mat'
            ],
            'sour': [
                'sour', 'acid', 'lemon', 'lime', 'vinegar', 'chua', 'chua',
                'chanh', 'giấm', 'giam', 'me', 'mẻ', 'chua cay', 'chua ngọt',
                'chua ngot', 'tamarind'
            ],
            'salty': [
                'salty', 'salt', 'savory', 'umami', 'mặn', 'man', 'muối', 'muoi',
                '짠', 'mặn mà', 'man ma', 'đậm đà', 'dam da', 'mặn ngọt', 'man ngot'
            ],
            'bitter': [
                'bitter', 'đắng', 'dang', 'khổ qua', 'kho qua', 'bitter melon',
                'coffee', 'cà phê', 'ca phe', 'dark chocolate'
            ],
            'mild': [
                'mild', 'nhẹ', 'nhe', 'thanh đạm', 'thanh dam', 'không cay', 'khong cay',
                'dịu', 'diu', 'nhạt', 'nhat', 'tươi mát', 'tuoi mat'
            ]
        }
        
        # Nguyên liệu chi tiết
        self.ingredient_keywords = {
            'chicken': [
                'chicken', 'poultry', 'hen', 'gà', 'ga', 'thịt gà', 'thit ga',
                'gà ta', 'ga ta', 'gà công nghiệp', 'ga cong nghiep', 'ức gà', 'uc ga',
                'đùi gà', 'dui ga', 'cánh gà', 'canh ga', 'gà rán', 'ga ran'
            ],
            'beef': [
                'beef', 'cow', 'steak', 'ground beef', 'bò', 'bo', 'thịt bò', 'thit bo',
                'bò tái', 'bo tai', 'bò chín', 'bo chin', 'bò viên', 'bo vien',
                'thăn bò', 'than bo', 'sườn bò', 'suon bo', 'bò kho', 'bo kho'
            ],
            'pork': [
                'pork', 'pig', 'bacon', 'ham', 'heo', 'lợn', 'lon', 'thịt heo', 'thit heo',
                'thịt lợn', 'thit lon', 'ba chỉ', 'ba chi', 'sườn heo', 'suon heo',
                'chân giò', 'chan gio', 'thịt xá xíu', 'thit xa xiu'
            ],
            'fish': [
                'fish', 'salmon', 'tuna', 'cod', 'seafood', 'cá', 'ca', 'cá hồi', 'ca hoi',
                'cá ngừ', 'ca ngu', 'cá thu', 'ca thu', 'cá chép', 'ca chep',
                'cá rô', 'ca ro', 'cá diêu hồng', 'ca dieu hong', 'cá basa', 'ca basa'
            ],
            'shrimp': [
                'shrimp', 'prawn', 'lobster', 'crab', 'tôm', 'tom', 'tôm càng', 'tom cang',
                'tôm sú', 'tom su', 'cua', 'cào cào', 'cao cao', 'tôm thẻ', 'tom the',
                'tôm tít', 'tom tit', 'nghêu', 'ngheu', 'sò', 'so'
            ],
            'vegetables': [
                'vegetables', 'veggie', 'carrot', 'broccoli', 'spinach', 'rau', 'rau củ', 'rau cu',
                'cà rót', 'ca rot', 'súp lơ', 'sup lo', 'rau bina', 'cải bó xôi', 'cai bo xoi',
                'cải thảo', 'cai thao', 'rau muống', 'rau muong', 'rau lang', 'đu đủ xanh',
                'du du xanh', 'bắp cải', 'bap cai', 'cà chua', 'ca chua', 'dưa chuột', 'dua chuot'
            ],
            'rice': [
                'rice', 'grain', 'jasmine rice', 'brown rice', 'cơm', 'com', 'gạo', 'gao',
                'cơm trắng', 'com trang', 'cơm tấm', 'com tam', 'cơm dẻo', 'com deo',
                'gạo tẻ', 'gao te', 'gạo nàng hương', 'gao nang huong', 'gạo ST25'
            ],
            'noodles': [
                'noodles', 'pasta', 'spaghetti', 'ramen', 'bún', 'bun', 'mì', 'mi',
                'phở', 'pho', 'miến', 'mien', 'bánh canh', 'banh canh', 'bánh phở', 'banh pho',
                'hủ tiếu', 'hu tieu', 'mì gói', 'mi goi', 'mì tôm', 'mi tom'
            ],
            'egg': [
                'egg', 'eggs', 'omelet', 'scrambled', 'trứng', 'trung', 'trứng gà', 'trung ga',
                'trứng vịt', 'trung vit', 'trứng cút', 'trung cut', 'trứng chiên', 'trung chien',
                'trứng luộc', 'trung luoc', 'trứng ốp la', 'trung op la'
            ],
            'tofu': [
                'tofu', 'soy', 'đậu hũ', 'dau hu', 'đậu phụ', 'dau phu', 'tàu hũ', 'tau hu',
                'đậu', 'dau', 'đậu xanh', 'dau xanh', 'đậu đỏ', 'dau do'
            ]
        }
        
        # Thời gian ăn
        self.meal_time_keywords = {
            'breakfast': [
                'breakfast', 'morning', 'brunch', 'early', 'sáng', 'sang', 'bữa sáng', 'bua sang',
                'điểm tâm', 'diem tam', 'ăn sáng', 'an sang', 'sáng sớm', 'sang som',
                'buổi sáng', 'buoi sang', 'bữa điểm tâm', 'bua diem tam'
            ],
            'lunch': [
                'lunch', 'noon', 'midday', 'afternoon', 'trưa', 'trua', 'bữa trưa', 'bua trua',
                'ăn trưa', 'an trua', 'giữa trưa', 'giua trua', 'buổi trưa', 'buoi trua',
                'cơm trưa', 'com trua'
            ],
            'dinner': [
                'dinner', 'evening', 'night', 'supper', 'tối', 'toi', 'bữa tối', 'bua toi',
                'ăn tối', 'an toi', 'buổi tối', 'buoi toi', 'cơm tối', 'com toi',
                'bữa chiều', 'bua chieu', 'chiều', 'chieu'
            ],
            'snack': [
                'snack', 'light meal', 'quick bite', 'appetizer', 'ăn vặt', 'an vat',
                'đồ ăn vặt', 'do an vat', 'nhâm nhi', 'nham nhi', 'ăn chơi', 'an choi',
                'bánh kẹo', 'banh keo', 'quà vặt', 'qua vat'
            ]
        }
        
        # Phương pháp nấu
        self.cooking_method_keywords = {
            'fried': ['chiên', 'chien', 'rán', 'ran', 'fried', 'deep fried', 'xào', 'xao'],
            'grilled': ['nướng', 'nuong', 'grilled', 'bbq', 'barbecue', 'nướng than', 'nuong than'],
            'boiled': ['luộc', 'luoc', 'boiled', 'luộc chín', 'luoc chin'],
            'steamed': ['hấp', 'hap', 'steamed', 'hấp chín', 'hap chin'],
            'soup': ['canh', 'súp', 'sup', 'soup', 'nước', 'nuoc', 'lẩu', 'lau'],
            'stir_fried': ['xào', 'xao', 'stir fried', 'rang', 'xào lăn', 'xao lan'],
            'braised': ['kho', 'braised', 'rim', 'om', 'ốm', 'niêu', 'nieu']
        }
        
        # Cửa hàng và loại hình
        self.restaurant_type_keywords = {
            'street_food': [
                'street food', 'food truck', 'roadside', 'đường phố', 'duong pho',
                'quán vỉa hè', 'quan via he', 'ăn vặt', 'an vat', 'hàng rong', 'hang rong'
            ],
            'restaurant': [
                'restaurant', 'nhà hàng', 'nha hang', 'quán ăn', 'quan an',
                'fine dining', 'sang trọng', 'sang trong'
            ],
            'fast_food': [
                'fast food', 'quick', 'nhanh', 'thức ăn nhanh', 'thuc an nhanh',
                'đồ ăn nhanh', 'do an nhanh'
            ],
            'home_cooking': [
                'home cooking', 'homemade', 'gia đình', 'gia dinh', 'nấu nhà', 'nau nha',
                'cơm nhà', 'com nha', 'tự nấu', 'tu nau'
            ]
        }

    def normalize_vietnamese_text(self, text: str) -> str:
        """Chuẩn hóa văn bản tiếng Việt"""
        if not isinstance(text, str):
            text = str(text)
        
        # Loại bỏ dấu thanh
        normalized = unidecode(text.lower())
        
        # Thay thế các ký tự đặc biệt tiếng Việt
        vietnamese_chars = {
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a', 'â': 'a', 'ầ': 'a', 'ấ': 'a',
            'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ê': 'e', 'ề': 'e', 'ế': 'e',
            'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o', 'ô': 'o', 'ồ': 'o', 'ố': 'o',
            'ộ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u', 'ư': 'u', 'ừ': 'u', 'ứ': 'u',
            'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y',
            'đ': 'd', 'Đ': 'd'
        }
        
        result = text.lower()
        for vn_char, replacement in vietnamese_chars.items():
            result = result.replace(vn_char, replacement)
        
        return result

    def extract_intent(self, user_input: str) -> Dict:
        """Trích xuất ý định từ câu hỏi của người dùng với độ chính xác cao"""
        if not isinstance(user_input, str):
            user_input = str(user_input)
        
        original_input = user_input.lower().strip()
        normalized_input = self.normalize_vietnamese_text(original_input)
        
        intent = {
            'cuisine': [],
            'dietary': [],
            'ingredients': [],
            'meal_time': [],
            'taste': [],
            'cooking_method': [],
            'restaurant_type': [],
            'raw_text': original_input,
            'normalized_text': normalized_input,
            'confidence': 0.0,
            'keywords_found': []
        }
        
        total_matches = 0
        found_keywords = []
        
        # Hàm helper để tìm keywords
        def find_keywords_in_category(category_dict, category_name):
            nonlocal total_matches, found_keywords
            
            for item, keywords in category_dict.items():
                for keyword in keywords:
                    normalized_keyword = self.normalize_vietnamese_text(keyword)
                    
                    # Kiểm tra trong cả text gốc và text chuẩn hóa
                    if (keyword in original_input or 
                        normalized_keyword in normalized_input or
                        keyword in normalized_input):
                        
                        if item not in intent[category_name]:
                            intent[category_name].append(item)
                            total_matches += 1
                            found_keywords.append(f"{category_name}:{item}:{keyword}")
                        break
        
        # Tìm tất cả các loại keywords
        find_keywords_in_category(self.cuisine_keywords, 'cuisine')
        find_keywords_in_category(self.dietary_keywords, 'dietary')
        find_keywords_in_category(self.ingredient_keywords, 'ingredients')
        find_keywords_in_category(self.meal_time_keywords, 'meal_time')
        find_keywords_in_category(self.taste_keywords, 'taste')
        find_keywords_in_category(self.cooking_method_keywords, 'cooking_method')
        find_keywords_in_category(self.restaurant_type_keywords, 'restaurant_type')
        
        intent['keywords_found'] = found_keywords
        
        # Tính confidence score nâng cao
        word_count = len(original_input.split())
        base_confidence = min(total_matches / max(word_count, 1), 1.0)
        
        # Bonus cho việc tìm thấy nhiều loại keywords
        category_bonus = len([cat for cat in ['cuisine', 'dietary', 'ingredients', 'meal_time', 'taste'] 
                             if intent[cat]]) * 0.1
        
        intent['confidence'] = min(base_confidence + category_bonus, 1.0)
        
        return intent

    def semantic_search(self, query: str, recipe_df: pd.DataFrame, top_k: int = 10) -> List[Dict]:
        """Tìm kiếm ngữ nghĩa với hỗ trợ tiếng Việt"""
        if not isinstance(recipe_df, pd.DataFrame) or recipe_df.empty:
            print("Lỗi: DataFrame công thức không hợp lệ hoặc rỗng")
            return []
        
        # Chuẩn bị text từ recipes
        recipe_texts = []
        for _, recipe in recipe_df.iterrows():
            text_parts = []
            for col in ['name', 'ingredients', 'tags', 'description']:
                if col in recipe and pd.notna(recipe[col]):
                    original_text = str(recipe[col])
                    normalized_text = self.normalize_vietnamese_text(original_text)
                    text_parts.extend([original_text, normalized_text])
            recipe_texts.append(' '.join(text_parts))
        
        # Chuẩn bị query
        normalized_query = self.normalize_vietnamese_text(query)
        combined_query = f"{query} {normalized_query}"
        
        # Tạo vectorizer với stop words tiếng Việt
        all_stopwords = list(self.vietnamese_stopwords) + ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        
        vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words=all_stopwords,
            ngram_range=(1, 3),
            lowercase=True,
            min_df=1,
            max_df=0.8,
            token_pattern=r'[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+'
        )
        
        try:
            all_texts = recipe_texts + [combined_query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            recipe_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, recipe_vectors).flatten()
            
            # Cải thiện scoring với weight cho các yếu tố quan trọng
            enhanced_scores = []
            intent = self.extract_intent(query)
            
            for idx, base_score in enumerate(similarities):
                recipe = recipe_df.iloc[idx]
                enhanced_score = base_score
                
                # Bonus cho exact name match
                recipe_name = str(recipe.get('name', '')).lower()
                if any(word in recipe_name for word in query.lower().split()):
                    enhanced_score += 0.3
                
                # Bonus cho cuisine match
                if intent['cuisine']:
                    recipe_text = ' '.join([
                        str(recipe.get('name', '')),
                        str(recipe.get('tags', '')),
                        str(recipe.get('ingredients', ''))
                    ]).lower()
                    
                    for cuisine in intent['cuisine']:
                        if cuisine in recipe_text or any(kw in recipe_text for kw in self.cuisine_keywords.get(cuisine, [])):
                            enhanced_score += 0.2
                            break
                
                # Bonus cho ingredient match
                if intent['ingredients']:
                    recipe_ingredients = str(recipe.get('ingredients', '')).lower()
                    for ingredient in intent['ingredients']:
                        if ingredient in recipe_ingredients or any(kw in recipe_ingredients for kw in self.ingredient_keywords.get(ingredient, [])):
                            enhanced_score += 0.15
                
                enhanced_scores.append(enhanced_score)
            
            # Sắp xếp theo enhanced score
            enhanced_scores = np.array(enhanced_scores)
            top_indices = enhanced_scores.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if enhanced_scores[idx] > 0.05:  # Threshold thấp hơn để có nhiều kết quả hơn
                    recipe = recipe_df.iloc[idx]
                    results.append({
                        'recipe_id': recipe.get('id', idx),
                        'name': recipe.get('name', ''),
                        'score': float(enhanced_scores[idx]),
                        'base_score': float(similarities[idx]),
                        'ingredients': recipe.get('ingredients', ''),
                        'tags': recipe.get('tags', ''),
                        'description': recipe.get('description', ''),
                        'cuisine': recipe.get('cuisine', ''),
                        'nutrition': recipe.get('nutrition', [0]),
                        'cooking_time': recipe.get('cooking_time', ''),
                        'difficulty': recipe.get('difficulty', '')
                    })
            
            return results
        except Exception as e:
            print(f"Lỗi khi thực hiện tìm kiếm ngữ nghĩa: {e}")
            return []

    def fuzzy_match_dishes(self, query: str, dish_names: List[str], threshold: int = 60) -> List[Tuple[str, int]]:
        """Tìm kiếm mờ cho tên món ăn với hỗ trợ tiếng Việt"""
        if not dish_names:
            print("Lỗi: Danh sách tên món ăn rỗng")
            return []
        
        try:
            # Chuẩn bị query variants
            original_query = query.lower()
            normalized_query = self.normalize_vietnamese_text(query)
            
            # Chuẩn bị dish name variants
            dish_variants = {}
            for dish in dish_names:
                original_dish = dish.lower()
                normalized_dish = self.normalize_vietnamese_text(dish)
                dish_variants[original_dish] = dish
                if normalized_dish != original_dish:
                    dish_variants[normalized_dish] = dish
            
            all_dish_names = list(dish_variants.keys())
            
            # Thực hiện fuzzy matching với các query variants
            matches_original = process.extract(original_query, all_dish_names, limit=15, scorer=fuzz.token_sort_ratio)
            matches_normalized = process.extract(normalized_query, all_dish_names, limit=15, scorer=fuzz.token_sort_ratio)
            
            # Kết hợp và loại bỏ trùng lặp
            all_matches = {}
            for match, score in matches_original + matches_normalized:
                original_dish_name = dish_variants[match]
                if original_dish_name not in all_matches or all_matches[original_dish_name] < score:
                    all_matches[original_dish_name] = score
            
            # Sắp xếp và lọc theo threshold
            sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
            return [(dish, score) for dish, score in sorted_matches if score >= threshold]
            
        except Exception as e:
            print(f"Lỗi khi thực hiện tìm kiếm mờ: {e}")
            return []

    def extract_cooking_preferences(self, user_input: str) -> Dict:
        """Trích xuất sở thích nấu ăn chi tiết"""
        intent = self.extract_intent(user_input)
        
        preferences = {
            'difficulty_level': 'any',  # easy, medium, hard, any
            'cooking_time': 'any',      # quick (<30min), medium (30-60min), long (>60min), any  
            'serving_size': 'any',      # 1-2, 3-4, 5+, any
            'budget': 'any',            # low, medium, high, any
            'equipment': [],            # oven, stove, microwave, etc.
            'dietary_restrictions': intent['dietary'],
            'preferred_cuisines': intent['cuisine'],
            'disliked_ingredients': [],
            'cooking_skills': 'beginner' # beginner, intermediate, advanced
        }
        
        # Phân tích thời gian nấu
        time_indicators = {
            'quick': ['nhanh', 'quick', 'fast', '15 phút', '20 phút', 'tốc hành', 'toc hanh'],
            'medium': ['vừa', 'vua', 'medium', '30 phút', '45 phút', '1 tiếng', 'bình thường', 'binh thuong'],
            'long': ['lâu', 'lau', 'slow', 'chậm', 'cham', '2 tiếng', 'cả ngày', 'ca ngay']
        }
        
        normalized_input = self.normalize_vietnamese_text(user_input.lower())
        
        for time_type, keywords in time_indicators.items():
            for keyword in keywords:
                if keyword in user_input.lower() or self.normalize_vietnamese_text(keyword) in normalized_input:
                    preferences['cooking_time'] = time_type
                    break
        
        # Phân tích độ khó
        difficulty_indicators = {
            'easy': ['dễ', 'de', 'easy', 'simple', 'đơn giản', 'don gian', 'cơ bản', 'co ban'],
            'medium': ['vừa', 'vua', 'medium', 'bình thường', 'binh thuong'],
            'hard': ['khó', 'kho', 'hard', 'difficult', 'phức tạp', 'phuc tap', 'chuyên nghiệp', 'chuyen nghiep']
        }
        
        for diff_type, keywords in difficulty_indicators.items():
            for keyword in keywords:
                if keyword in user_input.lower() or self.normalize_vietnamese_text(keyword) in normalized_input:
                    preferences['difficulty_level'] = diff_type
                    break
        
        return preferences

    def suggest_recipes_by_mood(self, mood: str) -> List[str]:
        """Gợi ý món ăn theo tâm trạng"""
        mood_to_food = {
            'happy': ['bánh ngọt', 'kem', 'pizza', 'hamburger', 'đồ chiên giòn'],
            'sad': ['cháo', 'súp', 'phở', 'bún riêu', 'đồ ăn ấm nóng'],
            'stressed': ['trà', 'chè', 'đồ ngọt nhẹ', 'salad', 'đồ thanh mát'],
            'energetic': ['đồ cay', 'lẩu', 'nướng', 'barbecue', 'đồ có nhiều protein'],
            'tired': ['đồ dễ nấu', 'mì gói', 'cơm hộp', 'đồ đông lạnh'],
            'romantic': ['đồ Ý', 'rượu vang', 'steak', 'chocolate', 'tráng miệng']
        }
        
        normalized_mood = self.normalize_vietnamese_text(mood.lower())
        
        # Tìm mood phù hợp
        for mood_key, suggestions in mood_to_food.items():
            if mood_key in normalized_mood or any(keyword in normalized_mood for keyword in [mood_key]):
                return suggestions
        
        return ['cơm', 'phở', 'bánh mì']  # default suggestions

    def analyze_user_query_complexity(self, query: str) -> Dict:
        """Phân tích độ phức tạp của câu hỏi người dùng"""
        intent = self.extract_intent(query)
        
        complexity_score = 0
        complexity_factors = []
        
        # Đếm số lượng tiêu chí
        criteria_count = sum([
            len(intent['cuisine']),
            len(intent['dietary']),
            len(intent['ingredients']),
            len(intent['meal_time']),
            len(intent['taste'])
        ])
        
        if criteria_count >= 3:
            complexity_score += 2
            complexity_factors.append('multiple_criteria')
        
        # Kiểm tra từ phủ định
        negative_words = ['không', 'khong', 'không muốn', 'khong muon', 'not', 'no', 'without']
        if any(word in query.lower() for word in negative_words):
            complexity_score += 1
            complexity_factors.append('negative_constraints')
        
        # Kiểm tra câu hỏi so sánh
        comparison_words = ['hơn', 'hon', 'better', 'vs', 'so với', 'so voi', 'thay vì', 'thay vi']
        if any(word in query.lower() for word in comparison_words):
            complexity_score += 1
            complexity_factors.append('comparison')
        
        # Độ dài câu
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 1
            complexity_factors.append('long_query')
        
        complexity_level = 'simple'
        if complexity_score >= 3:
            complexity_level = 'complex'
        elif complexity_score >= 1:
            complexity_level = 'medium'
        
        return {
            'complexity_level': complexity_level,
            'complexity_score': complexity_score,
            'factors': complexity_factors,
            'criteria_count': criteria_count,
            'word_count': word_count,
            'intent': intent
        }

if __name__ == "__main__":
    # Test NLPProcessor với nhiều test case tiếng Việt
    try:
        nlp_processor = NLPProcessor()
        print("🎉 Khởi tạo NLPProcessor thành công!\n")
        
        # Test cases tiếng Việt
        test_queries = [
            "Tôi muốn ăn phở bò không cay cho bữa sáng",
            "Tìm món ăn chay Ý với mì pasta",
            "Món nướng cay cho buổi tối",
            "Đồ ăn nhanh ít calo cho người tập gym",
            "Bánh ngọt tiếng Pháp dễ làm",
            "Lẩu Thái chua cay nhiều rau",
            "Cơm tấm sườn nướng miền Nam",
            "Món Nhật thanh đạm cho người ăn kiêng"
        ]
        
        print("=== TEST INTENT EXTRACTION ===")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            intent = nlp_processor.extract_intent(query)
            print(f"   Cuisine: {intent['cuisine']}")
            print(f"   Dietary: {intent['dietary']}")
            print(f"   Ingredients: {intent['ingredients']}")
            print(f"   Taste: {intent['taste']}")
            print(f"   Meal time: {intent['meal_time']}")
            print(f"   Confidence: {intent['confidence']:.2f}")
            
            # Test complexity analysis
            complexity = nlp_processor.analyze_user_query_complexity(query)
            print(f"   Complexity: {complexity['complexity_level']} (score: {complexity['complexity_score']})")
        
        print("\n=== TEST FUZZY MATCHING ===")
        sample_dishes = [
            "Phở Bò Tái", "Bánh Mì Thịt Nướng", "Cơm Tấm Sườn", "Bún Bò Huế",
            "Bánh Xèo Miền Tây", "Gỏi Cuốn Tôm Thit", "Chả Cá Lã Vọng", 
            "Bún Chả Hà Nội", "Cao Lầu Hội An", "Mì Quảng Đà Nẵng"
        ]
        
        fuzzy_tests = ["pho bo", "banh mi", "com tam", "bun bo hue"]
        for test_query in fuzzy_tests:
            fuzzy_results = nlp_processor.fuzzy_match_dishes(test_query, sample_dishes, threshold=50)
            print(f"\nFuzzy search cho '{test_query}':")
            for dish, score in fuzzy_results[:3]:
                print(f"  - {dish}: {score}%")
        
        print("\n=== TEST SEMANTIC SEARCH ===")
        # Tạo sample DataFrame với nhiều món ăn Việt Nam
        sample_data = pd.DataFrame({
            'id': range(1, 11),
            'name': [
                "Phở Bò Tái Chín", "Bánh Mì Pate", "Cơm Tấm Sườn Nướng", 
                "Bún Bò Huế", "Bánh Xèo Tôm Thịt", "Gỏi Cuốn Chay",
                "Chả Cá Thăng Long", "Bún Chả Hà Nội", "Cao Lầu Hội An", "Mì Quảng Gà"
            ],
            'ingredients': [
                "bánh phở, thịt bò, hành tây, ngò", "bánh mì, pate, rau sống, dưa chua",
                "cơm tấm, sườn nướng, chả, bì", "bún, thịt bò, chả cua, tôm",
                "bột bánh xèo, tôm, thịt, giá đỗ", "bánh tráng, rau sống, bún, đậu hũ",
                "cá lăng, thịt nướng, bún, rau thơm", "bún, thịt nướng, chả, rau",
                "mì cao lầu, thịt heo, tôm khô", "mì quảng, gà, tôm, trứng cút"
            ],
            'tags': [
                "vietnamese, soup, beef", "vietnamese, sandwich, breakfast",
                "vietnamese, rice, grilled", "vietnamese, spicy, soup",
                "vietnamese, pancake, crispy", "vietnamese, vegetarian, fresh",
                "vietnamese, fish, noodles", "vietnamese, grilled, hanoi",
                "vietnamese, noodles, hoian", "vietnamese, noodles, chicken"
            ],
            'description': [
                "Món phở truyền thống Hà Nội", "Bánh mì Sài Gòn thơm ngon",
                "Cơm tấm Sài Gòn đặc sản", "Bún bò Huế cay nồng",
                "Bánh xèo miền Tây giòn rụm", "Gỏi cuốn chay thanh mát",
                "Chả cá Hà Nội truyền thống", "Bún chả Obama nổi tiếng",
                "Cao lầu đặc sản Hội An", "Mì Quảng đậm đà hương vị"
            ],
            'cuisine': ["vietnamese"] * 10,
            'cooking_time': ["30 phút", "15 phút", "45 phút", "60 phút", "30 phút", 
                           "20 phút", "40 phút", "35 phút", "25 phút", "50 phút"],
            'difficulty': ["medium", "easy", "medium", "hard", "medium", 
                          "easy", "medium", "medium", "easy", "medium"]
        })
        
        semantic_test_queries = [
            "món phở bò cho bữa sáng",
            "đồ ăn chay nhẹ nhàng", 
            "món nướng thơm ngon",
            "bún tôm chua cay"
        ]
        
        for query in semantic_test_queries:
            print(f"\nSemantic search cho: '{query}'")
            results = nlp_processor.semantic_search(query, sample_data, top_k=3)
            for result in results:
                print(f"  - {result['name']}: {result['score']:.3f}")
                print(f"    Ingredients: {result['ingredients'][:50]}...")
        
        print("\n=== TEST COOKING PREFERENCES ===")
        pref_queries = [
            "Tôi muốn nấu món dễ và nhanh dưới 30 phút",
            "Tìm món khó và phức tạp cho 4 người",
            "Đồ ăn chay đơn giản cho người mới học nấu"
        ]
        
        for query in pref_queries:
            prefs = nlp_processor.extract_cooking_preferences(query)
            print(f"\nPreferences cho: '{query}'")
            print(f"  - Difficulty: {prefs['difficulty_level']}")
            print(f"  - Time: {prefs['cooking_time']}")
            print(f"  - Dietary: {prefs['dietary_restrictions']}")
        
        print("\n Tất cả test cases đã hoàn thành thành công!")
        
    except Exception as e:
        print(f" Lỗi khi kiểm tra: {e}")
        import traceback
        traceback.print_exc()