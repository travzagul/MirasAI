import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MirasAI Akıllı Rehber", layout="centered", page_icon="🕌")

st.title("🕌 MirasAI: Akıllı Turizm ve Rota Asistanı")
st.markdown("İstanbul'un tarihi güzelliklerini keşfet ve sana en uygun rotayı oluştur!")

# --- 1. SINIF VE İSİM SÖZLÜKLERİ ---
SINIFLAR = [
    'Ayasofya_Camii', 'Beylerbeyi_Sarayi', 'Dolmabahce_Sarayi', 'Galata_Kulesi', 
    'Haydarpasa_Gari', 'Kiz_Kulesi', 'Ortakoy_Cami', 'Rumeli_Hisari', 
    'Suleymaniye_Camii', 'Sultan_Ahmet_Camii', 'Topkapi_Sarayi', 'Yerebatan_Sarnici'
]

# TURKCE_ISIMLER sözlüğünü de bu sıraya göre ve doğru isimlerle güncelledim
TURKCE_ISIMLER = {
    'Ayasofya_Camii': 'Ayasofya Camii', 
    'Beylerbeyi_Sarayi': 'Beylerbeyi Sarayı', 
    'Dolmabahce_Sarayi': 'Dolmabahçe Sarayı', 
    'Galata_Kulesi': 'Galata Kulesi', 
    'Haydarpasa_Gari': 'Haydarpaşa Garı', 
    'Kiz_Kulesi': 'Kız Kulesi', 
    'Ortakoy_Cami': 'Ortaköy Camii', 
    'Rumeli_Hisari': 'Rumeli Hisarı', 
    'Suleymaniye_Camii': 'Süleymaniye Camii', 
    'Sultan_Ahmet_Camii': 'Sultan Ahmet Camii', 
    'Topkapi_Sarayi': 'Topkapı Sarayı', 
    'Yerebatan_Sarnici': 'Yerebatan Sarnıcı'
}

@st.cache_resource
def modeli_yukle():
    # Eğitimdeki yapının aynısı: DEFAULT ağırlıklarla başlatıyoruz
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Çıkış katmanını eğitimdeki ile aynı yapıyoruz (12 mekan sınıfı)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(SINIFLAR))
    
    # Kendi eğittiğin ağırlıkları üzerine yüklüyoruz
    # map_location='cpu' olması çok önemli, çünkü Hugging Face'te GPU olmayabilir
    model.load_state_dict(torch.load('miras_ai_resnet18_weights.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model


# --- VERİ TEMİZLEME (PYARROW HATASI GİDERİLDİ) ---
def gecerli_mekan_mi(mekan_adi):
    ad = str(mekan_adi)
    if ad.isnumeric(): return False
    if re.search(r'[\u0600-\u06FF\u0400-\u04FF]', ad): return False
    return True

@st.cache_data
def veri_yukle():
    df = pd.read_csv('master_lokasyonlar.csv')
    df = df[df['Mekan_Adi'].apply(gecerli_mekan_mi)]
    if 'Kategori' not in df.columns:
        df['Kategori'] = 'Turistik'
        ekstra = pd.DataFrame([
            {'Mekan_Adi': 'IBB Haliç Sosyal Tesisleri', 'Enlem': 41.0250, 'Boylam': 28.9600, 'Kategori': 'Sosyal Tesis'},
            {'Mekan_Adi': 'Beltur Gülhane', 'Enlem': 41.0136, 'Boylam': 28.9800, 'Kategori': 'Sosyal Tesis'},
            {'Mekan_Adi': 'Galata Konak Cafe', 'Enlem': 41.0265, 'Boylam': 28.9740, 'Kategori': 'Sosyal Tesis'}
        ])
        df = pd.concat([df, ekstra], ignore_index=True)
    return df

model = modeli_yukle()
df = veri_yukle()

# --- GELİŞMİŞ FONKSİYONLAR ---
def metin_temizle(metin):
    metin = str(metin).lower()
    for tr, en in {'ı': 'i', 'i̇': 'i', 'ö': 'o', 'ü': 'u', 'ç': 'c', 'ş': 's', 'ğ': 'g'}.items():
        metin = metin.replace(tr, en)
    return metin.replace('_', ' ').strip()

def fotograf_analiz(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # EĞER BU SATIR YOKSA MUTLAKA EKLE:
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, 2)
    
    tahminler = []
    for i in range(2):
        isim = SINIFLAR[top_indices[0][i].item()]
        tahminler.append((TURKCE_ISIMLER.get(isim, isim.replace('_', ' ')), top_probs[0][i].item()))
    return tahminler


def mesafe_hesapla(enlem1, boylam1, enlem2, boylam2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [enlem1, boylam1, enlem2, boylam2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def onerileri_getir(hedef_mekan, df):
    df_temp = df.copy()
    aranan_ad = metin_temizle(hedef_mekan)
    df_temp['Arama_Ad'] = df_temp['Mekan_Adi'].apply(metin_temizle)
    secilen = df_temp[df_temp['Arama_Ad'].str.contains(aranan_ad, case=False, na=False)]
    if secilen.empty: return None, None
    enlem, boylam = secilen.iloc[0]['Enlem'], secilen.iloc[0]['Boylam']
    df_temp['Mesafe'] = df_temp.apply(lambda row: mesafe_hesapla(enlem, boylam, row['Enlem'], row['Boylam']), axis=1)
    df_temp = df_temp[df_temp['Arama_Ad'] != aranan_ad].sort_values('Mesafe')
    turistik = df_temp[(df_temp['Kategori'].str.contains('Turistik', case=False, na=False)) & (df_temp['Mesafe'] <= 3.0)]
    sosyal = df_temp[df_temp['Kategori'].str.contains('Sosyal', case=False, na=False)].head(3)
    return turistik, sosyal

# --- ARAYÜZ (SEKMELER) ---
tab1, tab2 = st.tabs(["📸 Fotoğraf ile Ara", "✍️ Manuel Yer Seç"])

with tab1:
    uploaded_file = st.file_uploader("Bir Mekan Fotoğrafı Yükle", type=["jpg", "png", "jpeg"])
    mekan_secimi = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
        tahminler = fotograf_analiz(image)
        
         # --- KESİN KARAR VE BELİRSİZLİK YÖNETİMİ ---
        en_yuksek_eminlik = tahminler[0][1]
        
        st.write("🧠 **Yapay Zeka'nın Analizi:**")
        
        if en_yuksek_eminlik < 0.50:
            # Eminlik %50'den düşükse kesinlikle reddet ve rota oluşturma
            st.error("⚠️ Alakasız Fotoğraf! Lütfen sisteme tanımlı İstanbul'a ait tarihi bir mekan fotoğrafı yükleyin.")
            st.info(f"🔍 Arka plan analizi: Sistem bu görseli en çok **{tahminler[0][0]}** mekanına benzetti ancak eminlik oranı yetersiz (%{en_yuksek_eminlik*100:.1f}).")
            
            # BURASI ÇOK ÖNEMLİ: None yaparak alttaki rota çıkarma kodlarının çalışmasını engelliyoruz
            mekan_secimi = None 
        else:
            # Eminlik %50'nin üzerindeyse tahmini gururla kabul et
            st.success(f"🥇 1. Tahmin: **{tahminler[0][0]}** (Eminlik: %{en_yuksek_eminlik*100:.1f})")
            if tahminler[1][1] > 0.10: 
                st.info(f"🥈 2. Tahmin: **{tahminler[1][0]}** (Eminlik: %{tahminler[1][1]*100:.1f})")
            
            mekan_secimi = tahminler[0][0]

    if mekan_secimi:
        turistik, sosyal = onerileri_getir(mekan_secimi, df)
        col1, col2 = st.columns(2)
        with col1:
            st.info("🏛️ Yakındaki Turistik Yerler (Maks 3 km)")
            if turistik is not None: st.table(turistik[['Mekan_Adi', 'Mesafe']])
        with col2:
            st.warning("☕ En Yakın 3 Sosyal Tesis")
            if sosyal is not None: st.table(sosyal[['Mekan_Adi', 'Mesafe']])

with tab2:
    st.subheader("Listeden Bir Mekan Seç")
    mekan_haritasi = {row['Mekan_Adi'].replace('_', ' '): row['Mekan_Adi'] for idx, row in df.iterrows() if 'Sosyal' not in str(row.get('Kategori', ''))}
    secilen = st.selectbox("Mekan:", list(mekan_haritasi.keys()))
    if st.button("Rotamı Oluştur"):
        turistik, sosyal = onerileri_getir(mekan_haritasi[secilen], df)
        c1, c2 = st.columns(2)
        c1.info("🏛️ Yakındaki Turistik Yerler"); c1.table(turistik[['Mekan_Adi', 'Mesafe']] if turistik is not None else "Bulunamadı")
        c2.warning("☕ En Yakın 3 Sosyal Tesis"); c2.table(sosyal[['Mekan_Adi', 'Mesafe']] if sosyal is not None else "Bulunamadı")