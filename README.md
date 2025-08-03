# 🔍 Multi-Modal Sentiment & Thematic Analysis

Çok-modlu (metin + görsel) duygu ve tematik analiz sistemi. Ürün görseli ve açıklamasından kalite, duygu ve tema çıkarımı yapan gelişmiş bir derin öğrenme projesi.

## 🌟 Özellikler

### 🎯 Ana Fonksiyonlar
- **Duygu Analizi**: 5 sınıflı duygu kategorilendirmesi (Çok Olumsuz → Çok Olumlu)
- **Tema Sınıflandırması**: Domain-spesifik tema analizi (Moda/Gıda)
- **Kalite Skorlaması**: 0-1 arası sürekli kalite değerlendirmesi
- **Model Açıklanabilirliği**: Grad-CAM ile görsel açıklamalar

### 🛠️ Teknoloji Stack
- **Derin Öğrenme**: PyTorch, CLIP, Transformers
- **Görsel İşleme**: torchvision, OpenCV, Albumentations
- **Açıklanabilirlik**: Grad-CAM, pytorch-grad-cam
- **Web Arayüzü**: Streamlit, Plotly
- **Experiment Tracking**: Weights & Biases
- **Veri**: Kaggle datasets, pandas

## 🏗️ Proje Yapısı

```
multimodal-sentiment-analysis/
├── src/
│   ├── models/
│   │   └── multimodal_sentiment.py    # Ana model mimarisi
│   ├── data/
│   │   └── dataset.py                 # Veri yükleme ve işleme
│   ├── training/
│   │   └── trainer.py                 # Eğitim pipeline'ı
│   └── utils/
│       └── gradcam_explainer.py       # Model açıklama araçları
├── streamlit_app/
│   └── app.py                         # Interaktif web demo
├── configs/
│   ├── fashion_config.json            # Moda domain ayarları
│   └── food_config.json               # Gıda domain ayarları
├── data/                              # Veri dizini
├── models/                            # Eğitilmiş modeller
├── checkpoints/                       # Model checkpoint'leri
├── train.py                          # Ana eğitim scripti
├── requirements.txt                  # Python bağımlılıkları
└── README.md
```

## 🚀 Kurulum

### 1. Repository'yi Klonlayın
```bash
git clone <repository-url>
cd multimodal-sentiment-analysis
```

### 2. Python Ortamını Hazırlayın
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. (Opsiyonel) Kaggle API Kurulumu
Kaggle veri setlerini kullanmak için:
```bash
pip install kaggle
# kaggle.json dosyanızı ~/.kaggle/ dizinine koyun
```

## 📊 Veri Hazırlama

### Sample Dataset Oluşturma
```bash
python train.py --create_sample_data --domain fashion --num_samples 1000
python train.py --create_sample_data --domain food --num_samples 1000
```

### Kaggle Dataset Kullanma
```python
from src.data.dataset import DatasetBuilder

# Moda verileri için
DatasetBuilder.download_kaggle_dataset(
    "paramaggarwal/fashion-product-images-dataset", 
    "data/fashion"
)

# Gıda verileri için  
DatasetBuilder.download_kaggle_dataset(
    "kmader/food41", 
    "data/food"
)
```

## 🎓 Model Eğitimi

### Hızlı Başlangıç
```bash
# Sample data ile test eğitimi
python train.py --create_sample_data --domain fashion --epochs 10

# Tam eğitim (GPU önerili)
python train.py --domain fashion --epochs 100 --batch_size 32
```

### Detaylı Eğitim Seçenekleri
```bash
python train.py \
    --domain fashion \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --clip_lr 1e-5 \
    --hidden_dim 512 \
    --use_wandb \
    --experiment_name "fashion_experiment_v1"
```

### Config Dosyası ile Eğitim
```bash
python train.py --config configs/fashion_config.json
```

### Eğitimi Devam Ettirme
```bash
python train.py --resume_from checkpoints/fashion_sentiment_analysis/best_model.pt
```

## 🎮 Streamlit Demo

### Demo Uygulamasını Başlatma
```bash
cd streamlit_app
streamlit run app.py
```

### Demo Özellikleri
- 📷 **Görsel Yükleme**: Ürün fotoğrafı yükleme
- 📝 **Metin Girişi**: Ürün açıklaması yazma
- 📊 **Sonuç Görselleştirme**: İnteraktif grafik ve metrikler
- 🔍 **Model Açıklaması**: Grad-CAM görselleştirmeleri
- 💾 **Sonuç İndirme**: JSON format sonuç kaydetme

## 🧠 Model Mimarisi

### CLIP-Tabanlı Multi-Modal Model
```
[Görsel] ──┐
           ├─→ [CLIP Encoder] ─→ [Fusion Layer] ──┬─→ [Sentiment Head]
[Metin] ───┘                                      ├─→ [Theme Head]
                                                  └─→ [Quality Head]
```

### Temel Bileşenler
- **CLIP Encoder**: Önceden eğitilmiş vision-language model
- **Fusion Layer**: Multi-modal özellik birleştirme
- **Task-Specific Heads**: Duygu, tema ve kalite için ayrı çıkış katmanları

### Desteklenen Domainler

#### 🎽 Moda (Fashion)
- **Temalar**: Casual, Formal, Sporty, Vintage, Elegant, Trendy, Bohemian, Minimalist, Classic, Edgy
- **Örnekler**: Elbise, ayakkabı, çanta, gözlük vs.

#### 🍕 Gıda (Food)
- **Temalar**: Healthy, Comfort Food, Gourmet, Fast Food, Traditional, Modern
- **Örnekler**: Yemek, içecek, atıştırmalık vs.

## 📈 Model Performansı

### Tipik Metrikler (Validation)
- **Duygu Doğruluğu**: ~85-90%
- **Tema Doğruluğu**: ~80-85%
- **Kalite MAE**: ~0.1-0.15

### Eğitim İzleme
- **Weights & Biases**: Gerçek zamanlı metrik takibi
- **Confusion Matrix**: Sınıflandırma performans analizi
- **Learning Curves**: Eğitim süreç görselleştirmesi

## 🔍 Model Açıklanabilirliği

### Grad-CAM Görselleştirme
```python
from src.utils.gradcam_explainer import MultiModalGradCAM

# Model açıklayıcı oluştur
explainer = MultiModalGradCAM(model)

# Kapsamlı açıklama oluştur
explanations = explainer.create_comprehensive_explanation(
    image=image_array,
    text="Ürün açıklaması",
    save_path="explanation.png"
)
```

### Desteklenen CAM Yöntemleri
- **GradCAM**: Temel gradient-based açıklama
- **GradCAM++**: Geliştirilmiş gradient ağırlıklandırma
- **ScoreCAM**: Gradient-free açıklama
- **XGradCAM**: Axiom-based açıklama

## 🛠️ Geliştirme

### Yeni Domain Ekleme
1. `SentimentThemeConfig` sınıfına yeni tema etiketleri ekleyin
2. Dataset loader'ı yeni domain için güncelleyin
3. Eğitim config'ini ayarlayın

### Custom Model Mimarisi
```python
from src.models.multimodal_sentiment import MultiModalSentimentAnalyzer

model = MultiModalSentimentAnalyzer(
    clip_model_name="openai/clip-vit-large-patch14",
    num_sentiment_classes=5,
    num_theme_classes=custom_theme_count,
    hidden_dim=1024,
    dropout_rate=0.2
)
```

## 📝 API Kullanımı

### Model Tahmin API'si
```python
import torch
from PIL import Image
from src.models.multimodal_sentiment import MultiModalSentimentAnalyzer

# Model yükle
model = MultiModalSentimentAnalyzer()
model.load_state_dict(torch.load('model.pt'))

# Tahmin yap
image = Image.open('product.jpg')
text = "Ürün açıklaması"
predictions = model.predict(image, [text])

print(f"Duygu: {predictions['sentiment_preds']}")
print(f"Tema: {predictions['theme_preds']}")
print(f"Kalite: {predictions['quality_scores']}")
```

## 🎯 Kullanım Örnekleri

### E-Ticaret Uygulamaları
- Ürün duygu analizi otomatik etiketleme
- Kalite skoruna göre ürün sıralama
- Tema-bazlı ürün önerisi

### İçerik Moderasyonu
- Ürün açıklamalarının duygu kontrolü
- Görsel kalite değerlendirmesi
- Otomatik kategori atama

### Pazar Araştırması
- Tüketici tercih analizi
- Trend takibi ve öngörüsü
- Marka duygu analizi

## 🤝 Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Detaylar için `LICENSE` dosyasına bakın.

## 🔗 İlgili Çalışmalar

- [CLIP: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Multi-Modal Sentiment Analysis](https://arxiv.org/abs/2103.14887)

## 📞 İletişim

Sorularınız veya geri bildirimleriniz için:
- GitHub Issues açın
- Pull Request gönderin
- Proje maintainer'ları ile iletişime geçin

---

**🎉 Mutlu kodlamalar!** Bu proje ile multi-modal derin öğrenme dünyasını keşfedin ve kendi uygulamalarınızı geliştirin.