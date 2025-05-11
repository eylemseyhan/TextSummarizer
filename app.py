# app.py
import streamlit as st
from summarizer import TextSummarizer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Metin Özetleyici", layout="wide")

st.title('Gelişmiş Metin Özetleyici')

# Initialize session state for model loading status
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Sidebar controls
with st.sidebar:
    st.header('Ayarlar')
    language = st.selectbox('Dil Seçimi', 
                          options=['turkish', 'english'],
                          format_func=lambda x: 'Türkçe' if x == 'turkish' else 'İngilizce',
                          help="Metninizin dilini seçin")
    
    use_transformers = st.checkbox('Transformer Model Kullan', value=True,
                                 help="Transformer modeller daha iyi anlama sağlar ancak daha yavaştır")
    
    n_sentences = st.slider('Özet Cümle Sayısı', 1, 10, 3,
                          help="Özette kaç cümle olmasını istiyorsunuz?")
    
    show_debug = st.checkbox('Detaylı Analiz Göster', value=False,
                           help="Özetleme süreci hakkında detaylı bilgi gösterir")
    
    st.markdown("""
    ### Hakkında
    Bu gelişmiş özetleyici, anlamlı özetler oluşturmak için ileri NLP teknikleri kullanır:
    - Kapsamlı ön işleme
    - Gelişmiş dil modelleri
    - Cümle önem puanlaması
    - Uzunluk bazlı filtreleme
    - Orijinal cümle sıralamasını koruma
    """)

# Initialize summarizer with loading indicator
if not st.session_state.model_loaded:
    with st.spinner('Dil modeli yükleniyor... İlk çalıştırmada bu işlem biraz zaman alabilir.'):
        @st.cache_resource
        def get_summarizer(lang, use_trans):
            return TextSummarizer(language=lang, use_transformers=use_trans)
        
        try:
            summarizer = get_summarizer(language, use_transformers)
            st.session_state.model_loaded = True
            st.session_state.summarizer = summarizer
        except Exception as e:
            st.error(f"Model yükleme hatası: {str(e)}")
            st.stop()
else:
    summarizer = st.session_state.summarizer

# Main content
col1, col2 = st.columns([6, 4])

with col1:
    st.header('Metin Girişi')
    text = st.text_area('Özetlemek istediğiniz metni buraya yazın:', height=300)

    if st.button('Özetle', type='primary'):
        if text:
            with st.spinner('Özetleniyor...'):
                result = summarizer.summarize_text(text, n_sentences=n_sentences, debug=show_debug)
                
                st.header('Özet')
                st.write(result['summary'])
                
                if show_debug and result['debug_info']:
                    with col2:
                        st.header('Analiz')
                        
                        # Preprocessing stats
                        st.subheader('Ön İşleme İstatistikleri')
                        stats = result['debug_info']['preprocessing_info']
                        st.metric("Orijinal Cümle Sayısı", stats['original_length'])
                        st.metric("Filtrelenmiş Cümle Sayısı", stats['filtered_length'])
                        st.metric("Kullanılan Model", 
                                "Transformer Model" if stats['model_type'] == 'Sentence Transformers' else "TF-IDF Model")
                        
                        # Sentence scores visualization
                        st.subheader('Cümle Önem Puanları')
                        scores = result['debug_info']['sentence_scores']
                        
                        # Convert to DataFrame for better visualization
                        df = pd.DataFrame({
                            'Cümle': [f"Cümle {i+1}" for i in range(len(scores))],
                            'Puan': list(scores.values())
                        })
                        
                        # Bar chart
                        fig = px.bar(df, x='Puan', y='Cümle', orientation='h',
                                   title='Cümle Önem Dağılımı')
                        fig.update_layout(
                            xaxis_title="Önem Puanı",
                            yaxis_title="Cümleler",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Score distribution
                        st.subheader('Puan Dağılımı')
                        fig2 = px.histogram(df, x='Puan', 
                                          title='Cümle Puanlarının Dağılımı')
                        fig2.update_layout(
                            xaxis_title="Önem Puanı",
                            yaxis_title="Cümle Sayısı"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Additional metrics
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Ortalama Puan", 
                                    f"{stats['average_score']:.3f}")
                        with col4:
                            st.metric("Seçilen Cümle Sayısı", 
                                    len(result['scores']))
        else:
            st.error('Lütfen özetlenecek bir metin girin.')

if not text:
    with col2:
        st.info('Analiz sonuçlarını burada görmek için sol panele metin girin.')
