from ultralytics import YOLO

def main():
    # 1. Memuat model dasar (Pre-trained) YOLOv8 khusus Klasifikasi Gambar
    # Kita pakai versi 'nano' (yolov8n-cls) supaya proses training-nya ringan & cepat di laptop
    model = YOLO('yolov8n-cls.pt')

    # 2. Mulai Proses Training (Tahap 3 di Panduan UAS)
    st_results = model.train(
        data='dataset',        # Mengarah ke folder utama 'dataset' yang berisi train, val, test
        epochs=30,             # Jumlah latihan (bisa diatur 10-20 untuk simulasi cepat)
        imgsz=48,              # Ukuran resolusi standar gambar FER-2013 (48x48 piksel)
        device='cpu'           # Gunakan 'cpu' jika laptop tidak memiliki kartu grafis/NVIDIA GPU
    )
    
    print("Training Selesai! Model terbaik Anda telah disimpan di folder: runs/classify/train/weights/best.pt")

if __name__ == '__main__':
    main()