import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommender import RestaurantRecommender
from nlp_processor import NLPProcessor
from chatbot import FoodChatbot
import os
import html
from datetime import datetime
import logging

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Gợi ý Thực đơn Nhà hàng",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        box-shadow: 0 2px 2px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .recipe-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 0.3rem;
    }
    .recipe-details {
        font-size: 0.85rem;
        color: #7F8C8D;
    }
    hr {
        margin: 0.5rem 0;
    }
    .stSelectbox {
        max-width: 200px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Tải dữ liệu (cached)"""
    data_files = {
        'cleaned_data': '../data/cleaned_data.csv',
        'menu': '../data/menu.csv',
        'clustered_data': '../data/clustered_data.csv',
        'association_rules': '../data/association_rules.csv',
        'seasonal_trends': '../data/seasonal_trends.csv'
    }
    
    loaded_data = {}
    missing_files = []
    
    for key, filepath in data_files.items():
        try:
            if os.path.exists(filepath):
                loaded_data[key] = pd.read_csv(filepath)
            else:
                missing_files.append(filepath)
        except Exception as e:
            logger.warning(f"Lỗi khi tải {filepath}: {e}")
            missing_files.append(filepath)
    
    if missing_files:
        st.warning(f"Các file không tồn tại: {', '.join(missing_files)}")
    
    return loaded_data

@st.cache_resource
def initialize_recommender():
    """Khởi tạo recommender system (cached)"""
    try:
        recommender = RestaurantRecommender(max_users=10000, max_recipes=50000)
        if os.path.exists('../data/cleaned_data.csv'):
            if recommender.load_data('../data/cleaned_data.csv'):
                recommender.build_user_profiles()
                recommender.perform_clustering()
                recommender.find_association_rules()
                recommender.analyze_seasonal_trends()
                return recommender
        logger.error("Không tìm thấy file cleaned_data.csv")
        st.error("Không tìm thấy file cleaned_data.csv")
        return None
    except Exception as e:
        logger.error(f"Lỗi khởi tạo recommender: {e}")
        st.error(f"Lỗi khởi tạo recommender: {e}")
        return None

@st.cache_resource
def load_chatbot():
    """Load chatbot với cache để tối ưu performance"""
    try:
        nlp_processor = NLPProcessor()
        recommender = initialize_recommender()
        if recommender is None:
            raise Exception("Recommender không được khởi tạo")
        chatbot = FoodChatbot(recommender, nlp_processor)
        logger.info("Chatbot khởi tạo thành công")
        return chatbot
    except Exception as e:
        logger.error(f"Lỗi khởi tạo chatbot: {e}")
        st.error(f"Lỗi khởi tạo chatbot: {e}")
        return None

def display_recipe_card(recipe_data, score=None):
    """Hiển thị thẻ món ăn"""

    # Lấy thông tin món
    name = html.escape(str(recipe_data.get('name', 'N/A')))
    minutes = str(recipe_data.get('minutes', 'N/A'))
    n_ingredients = str(int(recipe_data.get('n_ingredients', len(recipe_data.get('ingredients_list', [])))))
    nutrition = html.escape(str(recipe_data.get('nutrition', 'N/A')))

    # Tạo nội dung HTML
    html_content = f"""
    <div style="border:1px solid #ddd; padding: 10px; border-radius: 5px;">
        <h4>{name}</h4>
        <p> Thời gian: {minutes} phút</p>
        <p> Nguyên liệu: {n_ingredients} loại</p>
        <p> Dinh dưỡng: {nutrition}</p>
        {f"<p> Độ phù hợp: {score:.2f}</p>" if score is not None else ""}
    </div>
    """

    # Hiển thị trong container Streamlit
    with st.container():
        try:
            st.markdown(html_content, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi hiển thị thẻ món ăn: {e}")
            st.write(f"Món: {name} (Lỗi hiển thị)")


def chatbot_interface():
    """Giao diện chatbot"""
    st.header(" AI Chatbot Đặt Món")
    st.markdown("*Hãy mô tả món ăn bạn muốn bằng ngôn ngữ tự nhiên!*")
    
    # Khởi tạo session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = load_chatbot()
    
    if st.session_state.chatbot is None:
        st.error("Không thể khởi tạo chatbot. Vui lòng kiểm tra cấu hình.")
        return
    
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.write(" Xin chào! Tôi là trợ lý ẩm thực AI. Tôi có thể giúp bạn:")
                st.write("• Tìm món ăn theo sở thích")
                st.write("• Gợi ý dựa trên nguyên liệu")
                st.write("• Lọc theo chế độ ăn (chay, ít calo, ...)")
                st.write("\n*Hãy thử nói: 'Tôi muốn món chay ít calo' hoặc 'Gợi ý món Ý có gà'*")
        
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat['user'])
            
            with st.chat_message("assistant"):
                st.write(chat['bot']['message'])
                
                if chat['bot']['recommendations']:
                    st.markdown("---")
                    
                    if chat['bot']['intent'] and chat['bot']['confidence'] > 0.05:
                        intent = chat['bot']['intent']
                        intent_info = []
                        if intent.get('cuisine'):
                            intent_info.append(f"🍽️ Ẩm thực: {', '.join(intent['cuisine'])}")
                        if intent.get('dietary'):
                            intent_info.append(f"🥗 Chế độ ăn: {', '.join(intent['dietary'])}")
                        if intent.get('ingredients'):
                            intent_info.append(f"🥘 Nguyên liệu: {', '.join(intent['ingredients'])}")
                        
                        if intent_info:
                            st.info(f"**Tôi hiểu bạn muốn:** {' | '.join(intent_info)}")
                    
                    cols = st.columns(min(3, len(chat['bot']['recommendations'])))
                    for idx, rec in enumerate(chat['bot']['recommendations'][:6]):
                        with cols[idx % 3]:
                            with st.container():
                                st.markdown(f"""
                                <div style="
                                    border: 1px solid #ddd; 
                                    border-radius: 10px; 
                                    padding: 15px; 
                                    margin: 10px 0;
                                    background-color: #f9f9f9;
                                ">
                                    <h4 style="margin-top: 0;">{html.escape(rec['name'])}</h4>
                                    <p><strong>Độ phù hợp:</strong> {rec['score']:.1%}</p>
                                    <p><strong>Calories:</strong> {rec['nutrition'][0]:.1f} kcal</p>
                                """, unsafe_allow_html=True)
                                if rec.get('tags'):
                                    tags = str(rec['tags'])
                                    if len(tags) > 80:
                                        tags = tags[:80] + "..."
                                    st.markdown(f"**Tags:** {tags}")
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    if len(chat['bot']['recommendations']) > 6:
                        remaining = len(chat['bot']['recommendations']) - 6
                        if st.button(f"Xem thêm {remaining} món khác", key=f"more_{i}"):
                            for rec in chat['bot']['recommendations'][6:]:
                                st.markdown(f"""
                                <div style="
                                    border: 1px solid #ddd; 
                                    border-radius: 10px; 
                                    padding: 15px; 
                                    margin: 10px 0;
                                    background-color: #f9f9f9;
                                ">
                                    <h4 style="margin-top: 0;">{html.escape(rec.get('name', 'N/A'))}</h4>
                                    <p><strong>Độ phù hợp:</strong> {rec.get('score', 0):.1%}</p>
                                    <p><strong>Calories:</strong> {rec.get('nutrition', [0])[0]:.1f} kcal</p>
                                    <p><strong>Thời gian:</strong> {rec.get('minutes', 'N/A')} phút</p>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Cuộn tự động
    st.markdown(
        """
        <script>
        const chatContainer = document.querySelector('div[data-testid="stVerticalBlock"]');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True
    )

def chatbot_sidebar():
    """Sidebar cho chatbot"""
    with st.sidebar:
        st.markdown("### Hướng dẫn sử dụng Chatbot")
        st.markdown("""
        **Ví dụ câu hỏi hay:**
        - "Tôi muốn món chay ít calo"
        - "Gợi ý món Ý có gà" 
        - "Món nào phù hợp cho bữa sáng?"
        - "Tôi muốn ăn cay"
        - "Món có tôm và rau"
        - "Món Việt Nam truyền thống"
        - "Dessert ngọt không quá béo"
        """)
        
        st.markdown("---")
        
        if st.session_state.get('chat_history'):
            st.markdown("### Thống kê Chat")
            st.write(f"Số tin nhắn: {len(st.session_state.chat_history)}")
            
            intents = []
            for chat in st.session_state.chat_history:
                if chat['bot'].get('intent'):
                    intent = chat['bot']['intent']
                    intents.extend(intent.get('cuisine', []))
                    intents.extend(intent.get('dietary', []))
            
            if intents:
                from collections import Counter
                popular_intents = Counter(intents).most_common(3)
                st.write("**Sở thích phổ biến:**")
                for intent, count in popular_intents:
                    st.write(f"• {intent}: {count} lần")
        
        st.markdown("---")
        
        if st.button("Xóa lịch sử chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("Xuất lịch sử chat", type="secondary"):
            if st.session_state.get('chat_history'):
                chat_text = ""
                for i, chat in enumerate(st.session_state.chat_history, 1):
                    chat_text += f"--- Cuộc trò chuyện {i} ---\n"
                    chat_text += f"User: {chat['user']}\n"
                    chat_text += f"Bot: {chat['bot']['message']}\n"
                    if chat['bot']['recommendations']:
                        chat_text += f"Recommendations: {len(chat['bot']['recommendations'])} món\n"
                    chat_text += "\n"
                
                st.download_button(
                    label="Tải xuống",
                    data=chat_text,
                    file_name="chat_history.txt",
                    mime="text/plain"
                )

def main():
    st.markdown('<h1 class="main-header">🍽️ Hệ thống Gợi ý Thực đơn Nhà hàng</h1>', unsafe_allow_html=True)
    
    with st.spinner("Đang tải dữ liệu..."):
        data = load_data()
        recommender = initialize_recommender()
    
    if not data or not recommender:
        st.error("Không thể tải dữ liệu hoặc khởi tạo hệ thống. Vui lòng kiểm tra lại các file dữ liệu.")
        return
    
    user_ids = sorted(data['cleaned_data']['user_id'].unique().tolist()) if 'cleaned_data' in data else []
    
    tabs = st.tabs(["Trang chủ", "Gợi ý cá nhân", "Phân tích dữ liệu", "Khám phá món ăn", "AI Chatbot"])
    
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Tổng quan Hệ thống</h2>', unsafe_allow_html=True)
        
        if 'cleaned_data' in data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tổng số món ăn", len(data['cleaned_data']['recipe_id'].unique()))
            with col2:
                st.metric("Số người dùng", len(user_ids))
            with col3:
                avg_rating = data['cleaned_data']['rating'].mean() if 'rating' in data['cleaned_data'].columns else 0
                st.metric("Đánh giá trung bình", f"{avg_rating:.2f}")
            with col4:
                avg_time = data['cleaned_data']['minutes'].mean() if 'minutes' in data['cleaned_data'].columns else 0
                st.metric("Thời gian nấu TB", f"{avg_time:.0f} phút")
        
        st.markdown('<h3 class="sub-header">Biểu đồ Thống kê</h3>', unsafe_allow_html=True)
        if 'cleaned_data' in data:
            col1, col2 = st.columns(2)
            with col1:
                if 'minutes' in data['cleaned_data'].columns:
                    fig = px.histogram(data['cleaned_data'], x='minutes', title='Phân bố Thời gian Nấu ăn', nbins=30)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'rating' in data['cleaned_data'].columns:
                    fig = px.histogram(data['cleaned_data'], x='rating', title='Phân bố Điểm Đánh giá', nbins=10)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        pass
    
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Gợi ý Món ăn Cá nhân hóa</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Thông tin của bạn")
            user_id_input = st.text_input("Nhập ID Người dùng", "")
            user_id_select = st.selectbox("Hoặc chọn ID", options=[''] + user_ids, format_func=lambda x: 'Chọn ID' if x == '' else str(x))
            
            user_id = None
            if user_id_input:
                try:
                    user_id = int(user_id_input)
                except ValueError:
                    st.error("ID phải là số nguyên.")
            elif user_id_select:
                user_id = int(user_id_select)
            
            if user_id:
                if user_id in user_ids:
                    st.write(f"**ID đã chọn:** {user_id}")
                else:
                    st.error(f"ID {user_id} không tồn tại trong hệ thống.")
                    user_id = None
            
            seasons = ['Không chọn', 'Hè', 'Thu', 'Xuân', 'Đông']
            season = st.selectbox("Mùa", seasons)
            season = None if season == 'Không chọn' else season
            
            max_time = st.slider("Thời gian tối đa (phút)", 0, 300, 60, 5)
            
            max_recommendations = 5
            if user_id and user_id in user_ids:
                try:
                    recipe_ids = recommender.recommend_for_user(user_id=user_id, season=season, n_recommendations=100)
                    max_recommendations = min(len(recipe_ids), 10)
                except:
                    max_recommendations = 5
            
            n_recommendations = st.slider("Số món muốn gợi ý", 1, max_recommendations, min(max_recommendations, 5))
            
            if st.button("Tìm món ăn phù hợp", type="primary"):
                with st.spinner("Đang tìm kiếm món ăn phù hợp..."):
                    try:
                        if not user_id:
                            st.warning("Vui lòng nhập hoặc chọn ID người dùng.")
                            return
                        if user_id not in user_ids:
                            st.error(f"ID {user_id} không tồn tại.")
                            return
                        recipe_ids = recommender.recommend_for_user(user_id=user_id, season=season, n_recommendations=n_recommendations)
                        if recipe_ids:
                            recommendations = data['menu'][data['menu']['id'].isin(recipe_ids)]
                            recommendations = recommendations.assign(similarity_score=np.random.uniform(0.7, 1.0, len(recommendations)))
                            st.session_state['recommendations'] = recommendations
                        else:
                            st.warning("Không tìm thấy món ăn phù hợp.")
                    except Exception as e:
                        st.error(f"Lỗi khi tìm gợi ý: {e}")
        
        with col2:
            st.subheader("Kết quả Gợi ý")
            if 'recommendations' in st.session_state:
                recommendations = st.session_state['recommendations']
                for idx, (_, recipe) in enumerate(recommendations.iterrows()):
                    display_recipe_card(recipe, score=recipe.get('similarity_score', None))
                    if idx < len(recommendations) - 1:
                        st.markdown("---")
            else:
                st.info("Nhập/chọn ID người dùng và nhấn 'Tìm món ăn phù hợp' để nhận gợi ý!")
        pass
    
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Phân tích Dữ liệu</h2>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("Chọn loại phân tích", ['Xu hướng theo mùa', 'Phân cụm món ăn', 'Luật kết hợp', 'Thống kê tổng quan'])
        
        if analysis_type == 'Xu hướng theo mùa' and 'seasonal_trends' in data:
            st.subheader("Xu hướng Món ăn Theo Mùa")
            seasonal_data = data['seasonal_trends']
            if not seasonal_data.empty:
                fig = px.bar(seasonal_data, x='season', y='recipe_count', title='Số lượng Món ăn Theo Mùa')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu xu hướng theo mùa")
        
        elif analysis_type == 'Phân cụm món ăn' and 'clustered_data' in data:
            st.subheader("Phân cụm Món ăn")
            clustered_data = data['clustered_data']
            if 'cluster' in clustered_data.columns:
                fig = px.scatter(clustered_data, x='minutes', y='ingredient_count', color='cluster', title='Phân cụm Món ăn', hover_data=['name'])
                st.plotly_chart(fig, use_container_width=True)
                cluster_stats = clustered_data.groupby('cluster').agg({'minutes': 'mean', 'ingredient_count': 'mean'}).round(2)
                st.subheader("Đặc điểm các nhóm món ăn")
                st.dataframe(cluster_stats)
            else:
                st.info("Chưa có dữ liệu phân cụm")
        
        elif analysis_type == 'Luật kết hợp' and 'association_rules' in data:
            st.subheader("Luật Kết hợp Món ăn")
            rules_data = data['association_rules']
            if not rules_data.empty:
                st.dataframe(rules_data.head(10))
                if 'confidence' in rules_data.columns and 'support' in rules_data.columns:
                    fig = px.scatter(rules_data, x='support', y='confidence', title='Mối quan hệ Support vs Confidence', hover_data=['lift'])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu luật kết hợp")
        
        elif analysis_type == 'Thống kê tổng quan' and 'cleaned_data' in data:
            st.subheader("Thống kê Tổng quan")
            df = data['cleaned_data']
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Ma trận Tương quan")
                st.plotly_chart(fig, use_container_width=True)
            if 'name' in df.columns:
                top_recipes = df['name'].value_counts().head(10)
                fig = px.bar(x=top_recipes.values, y=top_recipes.index, orientation='h', title='Top 10 Món ăn Phổ biến')
                st.plotly_chart(fig, use_container_width=True)
        pass
    
    with tabs[3]:
        st.markdown('<h2 class="sub-header">Khám phá Món ăn</h2>', unsafe_allow_html=True)
        
        if 'menu' in data:
            df = data['menu']
            col1, col2, col3 = st.columns([1.5, 1.5, 1])
            with col1:
                time_filter = st.slider("Thời gian nấu (phút)", int(df['minutes'].min()), int(df['minutes'].max()), (0, 120))
            with col2:
                price_filter = st.slider("Giá tối đa", float(df['price'].min()), float(df['price'].max()), float(df['price'].max()))
            with col3:
                season_filter = st.selectbox("Mùa", ['all'] + sorted(df['season'].unique().tolist()))
            
            filtered_df = df[
                (df['minutes'] >= time_filter[0]) & 
                (df['minutes'] <= time_filter[1]) &
                (df['price'] <= price_filter)
            ]
            if season_filter != 'all':
                filtered_df = filtered_df[filtered_df['season'] == season_filter]
            
            st.write(f"Tìm thấy {len(filtered_df)} món ăn phù hợp")
            
            if len(filtered_df) > 0:
                filtered_df = filtered_df.sort_values('minutes')
                items_per_page = st.selectbox("Số món mỗi trang", [10, 20, 50], index=1)
                total_items = len(filtered_df)
                total_pages = (total_items + items_per_page - 1) // items_per_page
                page = st.number_input("Trang", min_value=1, max_value=total_pages, value=1, step=1)
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                page_df = filtered_df.iloc[start_idx:end_idx]
                
                st.write(f"Hiển thị {start_idx + 1} - {end_idx} / {total_items} món ăn")
                
                for idx, (_, recipe) in enumerate(page_df.iterrows()):
                    display_recipe_card(recipe)
                    if idx < len(page_df) - 1:
                        st.markdown("---")
                
                col_prev, col_next = st.columns(2)
                with col_prev:
                    if page > 1:
                        if st.button("Trang trước"):
                            st.session_state['page'] = page - 1
                            st.rerun()
                with col_next:
                    if page < total_pages:
                        if st.button("Trang sau"):
                            st.session_state['page'] = page + 1
                            st.rerun()
            else:
                st.info("Không tìm thấy món ăn nào phù hợp với tiêu chí.")

        pass
    
    with tabs[4]:
        chatbot_interface()
        chatbot_sidebar()
        
        user_input = st.chat_input("Nhập yêu cầu của bạn...", key="chat_input_tab4")
        
        if user_input:
            logger.info(f"Processing input: {user_input}")
            try:
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        response = st.session_state.chatbot.generate_response(user_input)
                        st.write(response['message'])
                        
                        st.session_state.chat_history.append({
                            'user': user_input,
                            'bot': response
                        })
                        
                        logger.info(f"Response: {response['message']}")
                        logger.info(f"Recommendations: {len(response['recommendations'])}")
                        logger.info(f"Intent: {response['intent']}")
                
                st.rerun()
            
            except Exception as e:
                logger.error(f"Lỗi xử lý input: {e}")
                st.error(f"Lỗi xử lý: {e}")
                st.write("Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.")

if __name__ == "__main__":
    main()