# app.py
import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Örüntü Tanıma: TextRank Özetleyici 📚✨", layout="centered")

st.title("📚 Örüntü Tanıma için TextRank Tabanlı Metin Özetleyici")
st.write("""
Bu proje, örüntü tanıma dersi kapsamında geliştirilmiştir.
Kullanıcıdan alınan serbest metin, cümlelere bölünür ve cümleler arası benzerlik grafı oluşturularak 
TextRank algoritmasıyla en önemli cümleler belirlenip özet oluşturulur.
""")

text_input = st.text_area("Metni buraya yapıştırın veya yazın:")

n_sentences = st.slider("Kaç cümlelik özet istiyorsun?", 1, 20, 5)

if st.button("Özetle"):
    if text_input.strip() == "":
        st.warning("Lütfen önce bir metin girin.")
    else:
        summary = summarize_text(text_input, n_sentences=n_sentences)
        st.subheader("Özet")
        st.write(summary)
