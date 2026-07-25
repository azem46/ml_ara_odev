"""
Kaan Hocam Merhabalar;

Projemin amacı : "Demografik Özellikler ile  Anksiyete, Algılanan Stres ve Uykusuzluk Düzeyi Ölçeklerinin Toplam
Puanları Kullanılarak Depresyon Şiddetinin Makine Öğrenmesi Algoritmaları ile Tahmin Edilmesi"dir.

Projemde kullandığım veri setini https://zenodo.org/records/10423537 adlı siteden temin ettim.
Veri setimden bahsedecek olursam bu veri seti, 27 Şubat - 17 Mart 2021 tarihleri ​​arasında 
toplanmış olup,demografik bilgileri, PHQ-9(Depresyon), GAD-7(Anksiyete), ISI(Uykusuzluk) ve PSS(Algılanan Stres) olmak üzere dört tanınmış psikolojik ölçeğe 
24.292 öğrencinin verdiği yanıtları içermektedir.

Veri seti toplam 5 ayrı tablodan oluşmaktadır. Tüm tablolarla "export_id" birincil anahtar bulunmaktadır.
Topla verilerimi bu değişken üzerinden birleştireceğim. 

Kısaca tablolar hakkında bilgi verecek olursam

*Demografik tabloda export_id değişkenin yanında cinsiyet,yaş,öğretinim durumu,sigara ve alkol kullanım bilgiler
*PHQ-9(Depresyon) tablosunda export_id değişkenin yanında toplam puan ve 9 farklı soruya verilen yanıt 
ve bu yanıtlara verilen cevap süreleri vardır.
*GAD-7(Anksiyete) tablosunda export_id değişkenin yanında toplam puan ve 7 farklı soruya verilen yanıt 
ve bu yanıtlara verilen cevap süreleri vardır.
*ISI(Uykusuzluk) tablosunda export_id değişkenin yanında toplam puan ve 7 farklı soruya verilen yanıt 
ve bu yanıtlara verilen cevap süreleri vardır.
*PSS(Algılanan Stres) tablosunda export_id değişkenin yanında toplam puan ve 14 farklı soruya verilen yanıt 
ve bu yanıtlara verilen cevap süreleri vardır.


Kendi veri setimi oluşturmak için ;
-İngilizce olan değişken adlarını ve değerlerini Türkçe'ye çevirdim.
-"export_id" deişkeni üzerinden 5 ayrı tabloyu birleştirdim. Birleştirirken demografik verilerin
yanına 4 ölçeğe ait toplam puanların yer aldığı değişkenleri ekledim. böylece elimde demografik
verilerle 4 farklı psikolojik teste ait toplam puan bilgisi oldu. Daha sonra "export_id" değişkenini
makine öğrenmesi aşamasında bir işime yaramayacağı için veri setimdem  çıkardım ve "depresyon_veri.csv" 
veri setimi oluşturdum. 

Kullanılan Kütüphaneler:
pip install pandas 
pip install numpy 
pip install scikit-learn
    
    
Çalışa Adımlarım:

    1.Kütüphanelerin eklenmesi
    2.Veri setinin yüklenmesi ve incelenmesi
    3.Eksik veri analizi
    4.Sayısal Sütünları ve Varsa Aykırı Değerleri Belirleyelim Sonrasında Temizleme
    5.Kategorik değişkenleri One-Hot Encoding yöntemiyle sayısal forma dönüştürme
    6.Sayısal değişkenlerde ölçekleme yapma.
    7."depresyon_puani" adlı sıralı değişkenden "depresyon_seviyesi" adlı 5 aşamalı
    kategorik değişken oluşturma (basit öz nitelik oluşturma)
    8. Hedef-target değişkeni veri setinden ayırma
    9. Veriyi train, validation ve test kümelerine ayırma ,stratify kullanma
    10.Logistik  Regresyon modelini eğitelim ve confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırma
    11.KNN modelini eğitelim ve confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırma
    12.Decision Tree modelini eğitelim ve confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırma 
    13.RandomForestClassifier modelini eğitelim ve confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırma 
    14.RidgeClassifier modelini eğitelim ve confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırma 




"""

#1. Kütüphanelerin Yüklenmesi
import pandas as pd
from sklearn.model_selection import train_test_split # eğitim ve test veri seti oluşturur
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

#2.Veri setinin yüklenmesi ve incelenmesi
#Verimizi daha iyi tanımak için ilk beş satırı ve sütunlara ve satırlara ait bilgileri gözlemliyoruz.
#satır ve sütun sayısını gözlemliyoruz.(24292, 9)
df = pd.read_csv("depresyon_veri.csv", index_col=False)

print(df.head())
print("------------------------------------------------")
print(df.shape)
print("------------------------------------------------")
print(df.info())

# 3. Eksik veri analizi (Yaptığımız incelemede değişkenlerimize ait eksik veri olmadığı görüyoruz)
print(df.isnull().sum())

#Sayısal Sütünları ve Varsa Aykırı Değerleri Belirleyelim (Hangi Sütunumuzda Kaç Tane Aykırı Değer Var Göreceğiz.)
#depresyon_puani değişenimde dönüşüm yapacağım için onda aykırı değer temziliği yapmıyorum.

sayisal_sutunlar = ["kaygi_puani","uykusuzluk_puani","stres_puani"]
aykiri_deger_maskesi = pd.Series(False, index = df.index)

for sutun in sayisal_sutunlar:

    q1 = df[sutun].quantile(0.25)
    q3 = df[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df[sutun] < alt_sinir) | (df[sutun] > ust_sinir)
    )


    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Sütun '{sutun}': Aykırı değer sayısı: {sutun_maskesi.sum()}")

#Aykırı değerlerimizi veri setinden çıkaralım.(2002 tane aykırı değer içeren satır çıkarıldı)

df_clean = df.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Aykırı değerler çıktıktan sonra \n{df_clean}")

#Kategorik değişkenleri One-Hot Encoding yöntemiyle sayısal forma dönüştürelim
kategorik_sutunlar = df_clean.select_dtypes(include=['object']).columns

df_encoded = pd.get_dummies(df_clean, columns=kategorik_sutunlar, drop_first=True)
print(df_encoded.head())
print(f"Yeni DataFrame'in şekli: {df_encoded.shape}")

#Sayısal değişkenlerde ölçekleme yapma.
scaler = StandardScaler()
sec_say_deg = ['kaygi_puani', 'uykusuzluk_puani', 'stres_puani']
df_encoded[sec_say_deg] = scaler.fit_transform(df_encoded[sec_say_deg])
print(df_encoded.head())

#Depresyon puanı 0-27 arasında bir puan , makine öğrenmemizin daha başarılı sonuç üretmesi için öz nitelik oluşturma
#olarak yeni bir değişken oluşturuyorum. depresyon_puanı adlı sıralı değişkenden depresyon seviyesi adlı 5 kategorili bir 
#değişken ürettim.
df_encoded['depresyon_seviyesi'] = df_encoded['depresyon_puani'].apply(
    lambda x: 0 if x <= 4 else (
              1 if x <= 9 else (
              2 if x <= 14 else (
              3 if x <= 19 else 4
              )
    )))

print(df_encoded.head(5))


#depresyon_puani değişkeniyle işimiz kalmadı onu veri setimizden çıkaralım.
df_encoded = df_encoded.drop('depresyon_puani', axis=1)

# Sütun kaldırıldıktan sonra DataFrame'in yeni şeklini gösterelim.
print(f"'depresyon_puani' sütunu kaldırıldıktan sonra DataFrame'in şekli: {df_encoded.shape}")

# Hedef değişkeni (y) ve özelliklerimizi (X) ayıralım
y = df_encoded['depresyon_seviyesi']
X = df_encoded.drop('depresyon_seviyesi', axis=1)
print(f"X'in şekli: {X.shape}")
print(f"y'nin şekli: {y.shape}")

#Verimizi  train, validation ve test kümelerine ayıralım

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # val = %80, test = %20
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42, stratify=y_train_val)

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

#Logistic regression modelinin tanımlanması ve eğitilmesi
log_reg_model = LogisticRegression(penalty="l2", C = 1, max_iter = 100,random_state=42,)
log_reg_model.fit(X_train, y_train)

print("Lojistik Regresyon Modeli Başarıyla Eğitildi!")

# Doğrulama seti üzerinde tahminler yapma
y_pred_val = log_reg_model.predict(X_val)

# Model performansını değerlendirme
print("\n--- Lojistik Regresyon Modeli Performansı (Doğrulama Seti) ---")
print("Doğruluk Skoru:", accuracy_score(y_val, y_pred_val))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred_val))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_val, y_pred_val))

# Karışıklık Matrisini görselleştirme
cm_log = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues',
            xticklabels=log_reg_model.classes_, yticklabels=log_reg_model.classes_)
plt.title('Karışıklık Matrisi (Logistik Regresyon - Orijinal Eğitim Verisiyle Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

#KNN modelinin tanımlanması , eğitilmesi ve sonuçlarının gözlemlenmesi
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
print("KNN Modeli Başarıyla Eğitildi!")

#Doğrulama seti üzerinde tahminler yapma (orijinal X_val kullanılarak)
y_pred_val_knn = knn_model.predict(X_val)

# Model performansını değerlendirme
print("\n--- KNN Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---")
print("Doğruluk Skoru:", accuracy_score(y_val, y_pred_val_knn))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred_val_knn))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_val, y_pred_val_knn))

# Karışıklık Matrisini görselleştirme
cm_knn = confusion_matrix(y_val, y_pred_val_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('Karışıklık Matrisi (KNN - Orijinal Eğitim Verisiyle Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

# Hyperparametre ayarlaması: ideal n_neighbors belirleme
k_accuracy = []
k_values = []
for k in range(3, 15):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    k_accuracy.append(accuracy_score(y_pred, y_test))
    k_values.append(k)

plt.plot(k_values, k_accuracy)
plt.show()

#Decision Tree modelinin tanımlanması , eğitilmesi ve sonuçlarının gözlemlenmesi
dtree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42, class_weight='balanced')

# Modeli orijinal eğitim verileriyle eğitme
dtree_model.fit(X_train, y_train)

print("Karar Ağacı Modeli Başarıyla Eğitildi!")

# Doğrulama seti üzerinde tahminler yapma (orijinal X_val kullanılarak)
y_pred_val_dtree = dtree_model.predict(X_val)

# Model performansını değerlendirme
print("\n--- Karar Ağacı Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---")
print("Doğruluk Skoru:", accuracy_score(y_val, y_pred_val_dtree))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred_val_dtree))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_val, y_pred_val_dtree))

# Karışıklık Matrisini görselleştirme
cm_dt = confusion_matrix(y_val, y_pred_val_dtree)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=dtree_model.classes_, yticklabels=dtree_model.classes_)
plt.title('Karışıklık Matrisi (Karar Ağacı Modeli - Orijinal Eğitim Verisiyle Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

#RandomForestClassifier modelinin tanımlanması , eğitilmesi ve sonuçlarının gözlemlenmesi
rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

# Modeli orijinal eğitim verileriyle eğitme
rf_model.fit(X_train, y_train)

print("Random Forest Modeli Başarıyla Eğitildi!")

# Doğrulama seti üzerinde tahminler yapma (orijinal X_val kullanılarak)
y_pred_val_rf = rf_model.predict(X_val)

# Model performansını değerlendirme
print("\n--- Random Forest Modeli Performansı (Orijinal Eğitim Verisiyle Doğrulama Seti) ---")
print("Doğruluk Skoru:", accuracy_score(y_val, y_pred_val_rf))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred_val_rf))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_val, y_pred_val_rf))

# Karışıklık Matrisini görselleştirme
cm_rf = confusion_matrix(y_val, y_pred_val_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Karışıklık Matrisi (Random Forest - Orijinal Eğitim Verisiyle Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

# Random Forest modelinden özellik önemlerini alalım(Hangi değişkenler daha etkili görelim)
feature_importances = rf_model.feature_importances_

# Özellik isimlerini (X_train sütun adları) alalım
feature_names = X_train.columns

# Özellik önemlerini bir DataFrame'e dönüştürelim
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Önem sırasına göre sıralayalım
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Random Forest Özellik Önemleri:")
print(importance_df)

# Özellik önemlerini görselleştirelim
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Random Forest Özellik Önemleri')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellik')
plt.tight_layout()
plt.show()

#RandomForestClassifier modelinin tanımlanması , eğitilmesi ve sonuçlarının gözlemlenmesi
# RidgeClassifier modelini başlatma
ridge_model = RidgeClassifier(random_state=42, class_weight='balanced')

# Modeli eğitim verileriyle eğitme
ridge_model.fit(X_train, y_train)
print("RidgeClassifier Modeli Başarıyla Eğitildi!")

# Doğrulama seti üzerinde tahminler yapma
y_pred_val_ridge = ridge_model.predict(X_val)

# Model performansını değerlendirme
print("\n--- RidgeClassifier Modeli Performansı (Doğrulama Seti) ---")
print("Doğruluk Skoru:", accuracy_score(y_val, y_pred_val_ridge))
print("\nSınıflandırma Raporu:\n", classification_report(y_val, y_pred_val_ridge))
print("\nKarışıklık Matrisi:\n", confusion_matrix(y_val, y_pred_val_ridge))

# Karışıklık Matrisini görselleştirme
cm_ridge = confusion_matrix(y_val, y_pred_val_ridge)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ridge, annot=True, fmt='d', cmap='Blues',
            xticklabels=ridge_model.classes_, yticklabels=ridge_model.classes_)
plt.title('Karışıklık Matrisi (RidgeClassifier - Doğrulama Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()


