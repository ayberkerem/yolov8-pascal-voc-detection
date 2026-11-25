# YOLOv8s Tabanlı Gerçek Zamanlı Günlük Nesne Tespiti

Bu proje, **Pascal VOC (2007 + 2012)** veri seti üzerinde ince ayar (fine-tuning) yapılmış bir **YOLOv8s** nesne tespiti modelini ve bu modeli interaktif olarak sunan **Gradio** web uygulamasını içerir.

## Proje Özeti
- **Görev:** Nesne Tespiti (Object Detection)
- **Model Mimarisi:** YOLOv8s (CSPDarknet + PANet + Anchor-Free Decoupled Head)
- **Veri Seti:** Pascal VOC 2007 + 2012 (20 sınıf; Train: 8.218, Val: 2.510, Test: 4.952 görsel)
- **Girdi Boyutu:** 640×640 RGB
- **Eğitim Ortamı:** Google Colab – NVIDIA Tesla T4 GPU
- **Eğitim Süresi:** 66,95 dakika (25 Epoch, Batch Size: 16, Optimizer: AdamW, lr0: 0.001)

## Test Kümesi Performansı (VOC 2007 Test)

| Metrik | Değer |
| :--- | :---: |
| **mAP@0.5** | `0.813` |
| **mAP@0.5:0.95** | `0.599` |
| **Precision** | `0.797` |
| **Recall** | `0.745` |
| **Inference Time (GPU)** | `7.9 ms` / görsel |

## Gerçek Dünya Testleri (5 Fotoğraf, Confidence = 0.35)
Model, telefon kamerasıyla çekilen 5 farklı ev içi fotoğrafta test edilmiştir:
- **Televizyon (`tvmonitor`):** `0.95` güven skoru ile doğru tespit (27.6 ms)
- **Cam Su Şişesi (`bottle`):** `0.92` güven skoru ile doğru tespit (27.6 ms)
- **Örtülü Koltuk (`sofa`):** `0.83` güven skoru ile doğru tespit (33.5 ms)
- **Saksı Bitkisi (`pottedplant`):** `0.79` güven skoru ile doğru tespit (19.5 ms)
- **Yemek Masası ve Sandalyeler:** `diningtable` (`0.62`) ve `chair` (`0.47`) tespit edildi; ters ışık ve kısmi kapanma nedeniyle bazı sandalyeler kaçırıldı ve bitişik kapitoneli sandalyeler yanlışlıkla `sofa` olarak etiketlendi (22.5 ms).

## Kurulum ve Çalıştırma

```bash
git clone [https://github.com/](https://github.com/)<ayberkerem>/yolov8-pascal-voc-detection.git
cd yolov8-pascal-voc-detection
pip install -r requirements.txt
python app.py