
# Hệ thống Gợi ý Thực đơn Nhà hàng

##  Mục tiêu dự án

Xây dựng hệ thống gợi ý thực đơn nhà hàng cá nhân hóa sử dụng Machine Learning, giúp khách hàng tìm được món ăn phù hợp với sở thích của mình.

##  Tính năng chính

###  Quy trình xử lý dữ liệu
- **Thu thập dữ liệu**: Sử dụng dataset từ Kaggle (RAW_recipes.csv, RAW_interactions.csv)
- **Làm sạch dữ liệu**: Loại bỏ outliers, xử lý missing values, chuẩn hóa định dạng
- **Tạo features**: Thời gian nấu, calories, số nguyên liệu, phân loại theo mùa

###  Machine Learning
- **Phân cụm K-Means**: Nhóm món ăn theo đặc điểm tương tự
- **Luật kết hợp Apriori**: Tìm mối quan hệ giữa các món ăn
- **Phân tích theo mùa**: Xu hướng món ăn theo thời gian

###  Giao diện người dùng
- **Streamlit Web App**: Giao diện thân thiện, dễ sử dụng
- **Gợi ý cá nhân hóa**: Dựa trên lịch sử đánh giá của user
- **Phân tích trực quan**: Biểu đồ, heatmap, metrics

## 🏗️ Cấu trúc dự án

```
Recommend_Food/
├── data/                          # Dữ liệu
│   ├── RAW_recipes.csv           # Dữ liệu gốc từ Kaggle
│   ├── RAW_interactions.csv      # Lịch sử tương tác
│   ├── cleaned_data.csv          # Dữ liệu đã làm sạch
│   ├── clustered_data.csv        # Dữ liệu sau phân cụm
│   ├── association_rules.csv     # Luật kết hợp
│   ├── seasonal_trends.csv       # Xu hướng theo mùa
│   └── menu.csv                  # Thực đơn cuối cùng
├── src/                          # Mã nguồn
│   ├── data_processing.py        # Xử lý dữ liệu
│   ├── recommender.py           # Hệ thống gợi ý
│   ├── check_data.py            # Kiểm tra dữ liệu
|   ├── nlp_processor.py               # Module xử lý ngôn ngữ tự nhiên (NLP)
|   ├── chatbot.py                     # Module chatbot tích hợp với giao diện web
│   └── app.py                   # Ứng dụng Streamlit
├── tests/                        # Unit tests
│   ├── test_chatbot.py
│   ├── test_data_processing.py
│   └── test_recommender.py
├── requirements.txt              # Dependencies
└── README.md                     # Tài liệu
```

##  Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
- Tải dataset từ Kaggle và đặt vào thư mục `data/`
- Đảm bảo có file `RAW_recipes.csv` và `RAW_interactions.csv`

### 3. Xử lý dữ liệu
```bash
cd src
python data_processing.py
```

### 4. Tạo model
```bash
python recommender.py
```

### 5. Kiểm tra dữ liệu
```bash
python check_data.py
```

### 6. Chạy ứng dụng
```bash
streamlit run app.py
```

## 🔧 Các thành phần chính

### DataProcessor (data_processing.py)
- Làm sạch dữ liệu recipes và interactions
- Xử lý nutrition, ingredients, time
- Tạo features mới cho ML

### RestaurantRecommender (recommender.py)
- Xây dựng user profiles từ lịch sử
- Phân cụm món ăn (K-Means)
- Tìm luật kết hợp (Apriori)
- Phân tích xu hướng theo mùa
- Tạo gợi ý cá nhân hóa

### Streamlit App (app.py)
- Giao diện web interactive
- Hiển thị gợi ý cá nhân hóa
- Phân tích dữ liệu trực quan
- Cấu hình tham số

##  Phương pháp gợi ý

### 1. Gợi ý dựa trên Cluster
- Phân tích sở thích user theo cluster
- Gợi ý món ăn từ cluster yêu thích

### 2. Gợi ý dựa trên Association Rules
- Tìm món ăn thường được đánh giá cao cùng nhau
- "Người thích A cũng thích B"

### 3. Gợi ý theo mùa
- Phân tích xu hướng theo thời gian
- Gợi ý món phù hợp với mùa hiện tại

##  Testing

Chạy unit tests:
```bash
cd tests
python -m pytest test_data_processing.py -v
python -m pytest test_recommender.py -v
```

##  Metrics đánh giá

- **Coverage**: Tỷ lệ món ăn được gợi ý
- **Diversity**: Đa dạng trong gợi ý
- **User Satisfaction**: Dựa trên rating feedback
- **Seasonal Relevance**: Phù hợp với mùa

##  Giao diện ứng dụng

### Trang chính
- Chọn User ID và tham số
- Hiển thị gợi ý cá nhân hóa
- Thông tin chi tiết món ăn

### Tab phân tích
- **Tổng quan**: Metrics và biểu đồ tổng thể
- **Phân cụm**: Kết quả K-Means clustering
- **Xu hướng mùa**: Heatmap và phân tích theo thời gian

##  Hướng phát triển

- [ ] Thêm

## Chatbox AI
## Bước 1: Cài đặt thư viện
pip install nltk spacy transformers torch fuzzywuzzy python-levenshtein

## Bước 2: Download spaCy model
python -m spacy download en_core_web_sm

## Bước 3: Tạo các file
1. Tạo src/nlp_processor.py
2. Tạo src/chatbot.py  
3. Cập nhật src/app.py
4. Tạo tests/test_chatbot.py

## Bước 4: Cập nhật requirements.txt
Thêm các thư viện mới vào requirements.txt

## Bước 5: Test
python -m pytest tests/test_chatbot.py -v

## Bước 6: Chạy ứng dụng
streamlit run src/app.py
'''

