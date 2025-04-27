# Örüntü Tanıma Dersi Projesi: TextRank Tabanlı Metin Özetleyici 

Bu proje, örüntü tanıma (Pattern Recognition) dersi kapsamında geliştirilmiştir.

## Amaç
Yüklenen serbest metin üzerinde:
- Cümlelere bölme (segmentation),
- Cümleler arası benzerlik ölçümü (cosine similarity),
- Benzerlik grafı kurma (Graph Construction),
- TextRank algoritmasıyla önemli cümleleri seçme (Pattern Detection),
- Ve özet oluşturma işlemleri yapılmaktadır.

## Kullanılan Yöntemler
- **TF-IDF Vektörleştirme:** Cümlelerin sayısal temsili için kullanılmıştır.
- **Cosine Similarity:** Cümleler arası benzerlik matrisi oluşturmak için kullanılmıştır.
- **NetworkX Graph:** Cümleler arasında ağırlıklı graph oluşturulmuştur.
- **PageRank (TextRank):** Önemli cümleler belirlenmiştir.
- **NLTK:** Cümlelere bölme işlemi gerçekleştirilmiştir.
- **Streamlit:** Kullanıcı arayüzü oluşturulmuştur.

## Kullanım
1. Streamlit uygulamasını başlat:
```bash
streamlit run app.py

 
