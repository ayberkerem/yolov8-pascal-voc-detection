import time
import cv2
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


def detect_objects(image, conf_threshold):
    if image is None:
        return None, "Lütfen bir görüntü yükleyin."

    # Gradio RGB verir; OpenCV/YOLO için BGR dönüşümü yapılır
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    start_t = time.perf_counter()
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        imgsz=640,
        verbose=False
    )
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    res = results[0]
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    counts = {}
    for box in res.boxes:
        cls_name = model.names[int(box.cls[0])]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    if counts:
        summary_lines = [f"- **{k}**: {v} adet" for k, v in counts.items()]
        summary = f"### Tespit Edilen Nesneler (Süre: {elapsed_ms:.1f} ms)\n" + "\n".join(summary_lines)
    else:
        summary = f"Eşik değeri ({conf_threshold}) üzerinde nesne tespit edilemedi. (Süre: {elapsed_ms:.1f} ms)"

    return annotated_rgb, summary


demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="numpy", label="Fotoğraf Yükle"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.35, step=0.05, label="Güven Eşiği (Confidence)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Model Çıktısı (Bounding Box)"),
        gr.Markdown(label="Tahmin Özeti")
    ],
    title="YOLOv8s – Pascal VOC Nesne Tespiti",
    description="Bir fotoğraf yükleyerek Pascal VOC (20 sınıf) üzerinde gerçek zamanlı nesne tespiti yapabilirsiniz."
)

if __name__ == "__main__":
    demo.launch()