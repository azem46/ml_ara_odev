"""
Merhabalar;

Bu proje makine öğrenmesi ara ödev kapsamında hazırlanmıştır. İlk uygulama ödevim.Bu nedenle eksik veya hatalı yanları olmasını gayet doğal karşılayacağınızı düşünüyorum. İlk amacım öğrenmek daha sonra da öğrendiklerimi göstermek. 

Veri setim ve oluşturulma şekli hakkında proje dosyasının en başında bilgi verdiğim için bu konuda yenileme yapmayacağım. Depresyon testinin sonucunu demografık değişkenler ve anksiyete, uykusuzluk ve stres testleriyle tahmin eden bir model geliştirmeye çalıştım. Bunu yaparken bir çok farklı model üzerinde çalıştım. Modellerin doğruluk derecelerini gözlemledim. Biraz bunlara değinecek olursam;

--- Lojistik Regresyon Modeli Performansı (Doğrulama Seti) ---
Doğruluk Skoru: 0.8292045141756125

--- KNN Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---
Doğruluk Skoru: 0.8114505917974126

---Karar Ağacı Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---
Doğruluk Skoru: 0.7396091384530691

--- Random Forest Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---
Doğruluk Skoru: 0.794522433250757

--- RidgeClassifier Modeli Performansı (Doğrulama Seti) ---
Doğruluk Skoru: 0.6911643270024773

şeklinde sonuçlar verdi. Hyperparametre ayarları bu oranları belli bir düzey daha artırabilirdi. 

Modellerimin başarılarını etkileyen en önemli faktör tahmin yani target değişkenimin (depresyon_seviyesi)dengesiz bir dağılım göstermesiydi. 

depresyon_seviyesi	
0-16766
1-5428
2-450
3-51
4-12

0 kategoride 16766 değer varken 4 kategorisinde 12 değer bulunuyordu. bu daha modellerimin performansını genel olarak olumsuz etkiledi.

Eğer buraya kadar geldiniz bana eşlik ettiyseniz teşekkür ederim. 

"""
