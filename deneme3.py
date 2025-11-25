import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import mysql.connector
import face_recognition
import threading
import time

# ====================== MySQL BAĞLANTISI ======================
DB_CFG = dict(
    host="localhost",
    user="root",
    password="gmz7042",  
    database="gamze_minikod"
)

try:
    db = mysql.connector.connect(**DB_CFG)
    cursor = db.cursor()
    print("Veritabani baglantisi basarili")
except mysql.connector.Error as e:
    print(f"Veritabani baglanti hatasi: {e}")
    db = None
    cursor = None

# ====================== KLASÖRLER VE DOSYALAR ======================
FOTO_KLASOR = "ogrenciler"
MODEL_DOSYA = "face_model.pkl"
os.makedirs(FOTO_KLASOR, exist_ok=True)

ENG2TR = {
    "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
    "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"
}

# Global değişkenler
known_face_encodings = []
known_face_names = []

# Kamera & Recognition
camera_active = False
camera_thread = None
current_frame = None
camera_label = None
camera_window = None
captured_frame = None

recognition_active = False
recognition_thread = None
recognition_frame = None
recognition_lock = threading.Lock()
last_recognized_name = "Bilinmiyor"
last_recognized_time = 0

# ====================== MODEL İŞLEMLERİ ======================
def model_egit_ve_kaydet():
    global known_face_encodings, known_face_names
    encodings, names = [], []
    for file in os.listdir(FOTO_KLASOR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(FOTO_KLASOR, file)
        try:
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(file)[0])
            else:
                print(f"Yüz bulunamadı: {file}")
        except Exception as e:
            print(f"Hata ({file}): {e}")
    
    if not encodings:
        raise RuntimeError("En az bir yüz fotoğrafı ekleyin.")
    
    known_face_encodings = encodings
    known_face_names = names
    
    with open(MODEL_DOSYA, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"Model kaydedildi: {len(names)} öğrenci")

def model_yukle():
    global known_face_encodings, known_face_names
    if os.path.exists(MODEL_DOSYA):
        try:
            with open(MODEL_DOSYA, "rb") as f:
                data = pickle.load(f)
            known_face_encodings = data.get("encodings", [])
            known_face_names = data.get("names", [])
            print(f"Model yüklendi: {len(known_face_names)} öğrenci")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            known_face_encodings, known_face_names = [], []
    else:
        print("Model yok, eğitiliyor...")
        try:
            model_egit_ve_kaydet()
        except Exception as e:
            print(f"Eğitim hatası: {e}")

# ====================== VERİTABANI FONKSİYONLARI ======================
def bugunun_tr_gunu():
    return ENG2TR.get(datetime.now().strftime("%A"), "Bilinmiyor")

def ogrenci_id_bul(ad, soyad):
    if not cursor: return None
    try:
        cursor.execute("SELECT id FROM ogrenciler WHERE ad=%s AND soyad=%s", (ad, soyad))
        r = cursor.fetchone()
        return r[0] if r else None
    except Exception as e:
        print(f"ID bulma hatası: {e}")
        return None

def yoklama_kaydi_ekle_ve_sayac(ogrenci_id, durum, tarih, saat):
    if not cursor or not db: return False
    try:
        cursor.execute("SELECT id, durum FROM yoklama WHERE ogrenci_id=%s AND tarih=%s", (ogrenci_id, tarih))
        existing = cursor.fetchone()
        if not existing:
            cursor.execute("INSERT INTO yoklama (ogrenci_id, tarih, durum, saat) VALUES (%s,%s,%s,%s)",
                          (ogrenci_id, tarih, durum, saat))
            db.commit()
            return True
        else:
            record_id, old_durum = existing
            if old_durum != durum:
                cursor.execute("UPDATE yoklama SET durum=%s, saat=%s WHERE id=%s", (durum, saat, record_id))
                db.commit()
                return True
        return False
    except Exception as e:
        print(f"Yoklama hatası: {e}")
        return False

# ====================== KAMERA & TANIMA ======================
def camera_thread_function(cap):
    global current_frame, camera_active, recognition_frame
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        # frame BGR, current_frame için RGB versiyonunu sakla
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Recognizer’a tam frame veriyoruz (BGR)
        with recognition_lock:
            recognition_frame = frame.copy()

        # Ekran için RGB tutuyoruz
        current_frame = frame_rgb
    cap.release()


def recognition_worker(mode="attendance"):
    global recognition_active, recognition_frame, current_frame, last_recognized_name, last_recognized_time
    import csv
    import time

    csv_file = "yoklama_log.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Isim", "Tarih", "Saat"])

    # --- Yüz Takip Sistemi ---
    tracked_faces = {}          
    next_face_id = 0
    max_distance = 80           # piksel, aynı yüz mü?
    max_missing_frames = 15     # 15 frame kaybolursa unut
    frame_count = 0
    recognition_interval = 3    # her 3 frame'de bir tanı (performans)

    while recognition_active:
        frame_for_recog = None
        with recognition_lock:
            if recognition_frame is not None:
                frame_for_recog = recognition_frame.copy()

        if frame_for_recog is None:
            time.sleep(0.05)
            continue

        frame_count += 1

        # face_recognition ile HOG modeli RGB bekler
        rgb_frame = cv2.cvtColor(frame_for_recog, cv2.COLOR_BGR2RGB)

        # Çizim için BGR 640x480 
        frame_draw = cv2.resize(frame_for_recog, (640, 480))

        current_faces = []
        matched_ids = set()

        # --- 1. Tanıma ---
        if frame_count % recognition_interval == 0:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2

                # --- Eşleştirme ---
                best_id = None
                best_dist = float('inf')
                best_name = "Bilinmiyor"

                # 1. Kayıtlı yüzlerle karşılaştır
                if known_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, encoding)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < 0.5:  # eşik
                        best_name = known_face_names[min_idx]

                        # Yoklama (sadece yeni tanıma)
                        now = time.time()
                        if (best_name != last_recognized_name) or (now - last_recognized_time > 3.0):
                            last_recognized_name = best_name
                            last_recognized_time = now
                            if mode == "attendance":
                                try:
                                    parts = best_name.split("_", 1)
                                    ad = parts[0]
                                    soyad = parts[1] if len(parts) > 1 else ""
                                    ogr_id = ogrenci_id_bul(ad, soyad)
                                    if ogr_id:
                                        dt = datetime.now()
                                        tarih = dt.strftime("%Y-%m-%d")
                                        saat = dt.strftime("%H:%M:%S")
                                        yoklama_kaydi_ekle_ve_sayac(ogr_id, "geldi", tarih, saat)
                                        with open(csv_file, "a", newline="", encoding="utf-8") as f:
                                            writer = csv.writer(f)
                                            writer.writerow([best_name, tarih, saat])
                                        print(f"{best_name} yoklaması alındı")
                                except Exception as e:
                                    print(f"Yoklama hatası: {e}")

                # 2. Takip edilen yüzlerle eşleştir
                for fid, (bbox, name, last_seen, enc) in tracked_faces.items():
                    if fid in matched_ids: continue
                    old_cx = (bbox[0] + bbox[2]) // 2
                    old_cy = (bbox[1] + bbox[3]) // 2
                    dist = ((center_x - old_cx)**2 + (center_y - old_cy)**2)**0.5
                    if dist < max_distance:
                        if dist < best_dist:
                            best_dist = dist
                            best_id = fid
                            best_name = name

                # --- Yeni veya güncellenmiş yüz ---
                if best_id is None:
                    best_id = next_face_id
                    next_face_id += 1

                tracked_faces[best_id] = (
                    (left, top, right, bottom),
                    best_name,
                    frame_count,
                    encoding
                )
                matched_ids.add(best_id)
                current_faces.append((left, top, right, bottom, best_name))

        # Eğer bu karede yüz yoksa, tracked_faces'tan daha önce görülenleri ekle
        if not current_faces and tracked_faces:
            for fid, (bbox, name, last_seen, enc) in tracked_faces.items():
                if frame_count - last_seen <= max_missing_frames:
                    current_faces.append((bbox[0], bbox[1], bbox[2], bbox[3], name))

        # Takip edilen ama bu frame'de görünmeyenleri temizle veya koru
        for fid, (bbox, name, last_seen, enc) in list(tracked_faces.items()):
            if fid not in matched_ids:
                if frame_count - last_seen <= max_missing_frames:
                    current_faces.append((bbox[0], bbox[1], bbox[2], bbox[3], name))
                else:
                    del tracked_faces[fid]

        # --- 3. Çizim: Sürekli kare ---
        h, w = frame_draw.shape[:2]
        for left, top, right, bottom, name in current_faces:
            # Koordinatları int'e çevir ve sınırla
            left = int(max(0, min(left, w-1)))
            top = int(max(0, min(top, h-1)))
            right = int(max(0, min(right, w-1)))
            bottom = int(max(0, min(bottom, h-1)))

            color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)
            thickness = 3

            cv2.rectangle(frame_draw, (left, top), (right, bottom), color, thickness)
            cv2.rectangle(frame_draw, (left, max(bottom - 35, top)), (right, bottom), color, -1)
            cv2.putText(frame_draw, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)


        # Yüz yoksa mesaj
        if not current_faces:
            cv2.putText(frame_draw, "Yuz algilanamadi", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        # --- Güncelle ---
        # frame_draw BGR, GUI için RGB'ye çevir
        frame_draw_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        with recognition_lock:
            current_frame = frame_draw_rgb.copy()

        time.sleep(0.25)  


def update_camera_display():
    global current_frame, camera_label, camera_active
    if camera_label and current_frame is not None:
        try:
            pil_image = Image.fromarray(current_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            camera_label.configure(image=tk_image)
            camera_label.image = tk_image
        except Exception as e:
            # hata görmezden gel
            print(f"Gosterim hatasi: {e}")
    if camera_active:
        # ~30 FPS ekran güncellemesi (yaklaşık)
        camera_label.after(33, update_camera_display)

def start_camera(mode="normal"):
    global camera_active, camera_thread, camera_label, camera_window
    global recognition_active, recognition_thread

    if camera_active:
        stop_camera()
        time.sleep(0.15)

    camera_window = tk.Toplevel()
    camera_window.title("Kamera")
    # Sabit boyut ver (küçülüp büyüme sorununu çözüyor)
    camera_window.geometry("680x520")
    camera_window.resizable(False, False)

    # Kamera görüntüsü için frame 
    top_frame = tk.Frame(camera_window)
    top_frame.pack(fill="both", expand=True, padx=8, pady=8)

    camera_label = tk.Label(top_frame, text="Kamera başlatılıyor...", bg="black", fg="white")
    camera_label.pack(fill="both", expand=True)

    control_frame = tk.Frame(camera_window)
    control_frame.pack(fill="x", padx=10, pady=6)

    if mode == "capture":
        tk.Button(control_frame, text="Fotoğraf Çek", command=capture_photo,
                  font=("Arial", 12, "bold"), width=12, height=1, bg="lightgreen").pack(side=tk.LEFT, padx=6)

    tk.Button(control_frame, text="Kamerayı Kapat", command=stop_camera,
              font=("Arial", 12, "bold"), width=12, height=1, bg="lightcoral").pack(side=tk.LEFT, padx=6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        camera_window.destroy()
        return None

    camera_active = True
    camera_thread = threading.Thread(target=camera_thread_function, args=(cap,), daemon=True)
    camera_thread.start()

    recognition_active = True
    recognition_thread = threading.Thread(target=recognition_worker, args=(mode,), daemon=True)
    recognition_thread.start()

    update_camera_display()

    def on_close():
        stop_camera()
        try:
            camera_window.destroy()
        except: pass
    camera_window.protocol("WM_DELETE_WINDOW", on_close)
    return cap

def stop_camera():
    global camera_active, recognition_active, camera_window
    camera_active = False
    recognition_active = False
    time.sleep(0.1)
    if camera_window:
        try:
            camera_window.destroy()
        except: pass

# ====================== FOTOĞRAF İŞLEMLERİ ======================
def capture_photo():
    global captured_frame, current_frame
    if current_frame is not None:
        captured_frame = current_frame.copy()
        messagebox.showinfo("Başarılı", "Fotoğraf çekildi!")
        return captured_frame
    return None

def process_captured_photo(ad, soyad):
    global captured_frame
    if captured_frame is None:
        messagebox.showerror("Hata", "Fotoğraf yok.")
        return
    frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
    safe_ad = ad.strip().replace(" ", "")
    safe_soyad = soyad.strip().replace(" ", "")
    foto_yolu = os.path.join(FOTO_KLASOR, f"{safe_ad}_{safe_soyad}.jpg")
    cv2.imwrite(foto_yolu, frame_bgr)
    try:
        if cursor and db:
            cursor.execute("UPDATE ogrenciler SET foto_yolu=%s WHERE ad=%s AND soyad=%s", (foto_yolu, ad, soyad))
            db.commit()
            model_egit_ve_kaydet()
            messagebox.showinfo("Başarılı", f"{ad} {soyad} eklendi.")
        else:
            messagebox.showwarning("Uyarı", "Sadece fotoğraf kaydedildi.")
    except Exception as e:
        messagebox.showwarning("Uyarı", f"Hata: {e}")
    captured_frame = None

# ====================== ÖĞRENCİ İŞLEMLERİ ======================
def yeni_ogrenci_kaydet():
    if not cursor or not db:
        messagebox.showerror("Hata", "Veritabanı yok!")
        return
    ad = simpledialog.askstring("Yeni Öğrenci", "Ad:")
    soyad = simpledialog.askstring("Yeni Öğrenci", "Soyad:")
    if not ad or not soyad: return

    gun_window = tk.Toplevel(root)
    gun_window.title("Ders Günleri")
    tk.Label(gun_window, text="Ders günlerini seç (Ctrl ile birden fazla)").pack(pady=5)
    listbox = tk.Listbox(gun_window, selectmode=tk.MULTIPLE)
    for gun in ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]:
        listbox.insert(tk.END, gun)
    listbox.pack(padx=10, pady=10)

    def kaydet():
        secilen = [listbox.get(i) for i in listbox.curselection()]
        if not secilen:
            messagebox.showerror("Hata", "Gün seçin!")
            return
        gun_str = ",".join(secilen)
        gun_window.destroy()
        messagebox.showinfo("Kamera", "Fotoğraf çekilecek...")
        cap = start_camera("capture")
        if not cap: return

        def check():
            if camera_window and camera_window.winfo_exists():
                root.after(200, check)
            else:
                if captured_frame is not None:
                    cursor.execute("INSERT INTO ogrenciler (ad, soyad, gun) VALUES (%s,%s,%s)", (ad, soyad, gun_str))
                    db.commit()
                    process_captured_photo(ad, soyad)
        root.after(200, check)
    tk.Button(gun_window, text="Kaydet", command=kaydet).pack(pady=5)

def yoklama_al():
    model_yukle()
    if not known_face_encodings:
        messagebox.showerror("Hata", "Öğrenci ekleyin.")
        return
    start_camera("attendance")
    messagebox.showinfo("Yoklama", f"YEŞİL = Kayıtlı\nKIRMIZI = Bilinmiyor\n{len(known_face_names)} öğrenci")

# ====================== RAPORLAMA ======================
def yoklamayi_excele_aktar():
    if not cursor: return
    cursor.execute("SELECT o.ad, o.soyad, y.tarih, y.durum FROM yoklama y JOIN ogrenciler o ON o.id = y.ogrenci_id ORDER BY y.tarih")
    rows = cursor.fetchall()
    if not rows:
        messagebox.showinfo("Bilgi", "Kayıt yok.")
        return
    df = pd.DataFrame(rows, columns=["Ad", "Soyad", "Tarih", "Durum"])
    dosya = f"yoklama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(dosya, index=False)
    messagebox.showinfo("Excel", f"Kaydedildi: {dosya}")

def yoklama_sayac_goster():
    if not cursor: return
    cursor.execute("SELECT o.ad, o.soyad, COUNT(y.id) FROM ogrenciler o LEFT JOIN yoklama y ON o.id = y.ogrenci_id GROUP BY o.id")
    rows = cursor.fetchall()
    win = tk.Toplevel(root)
    win.title("Yoklama Sayacı")
    text = tk.Text(win)
    for ad, soyad, sayi in rows:
        text.insert(tk.END, f"{ad} {soyad}: {sayi} gün\n")
    text.pack()

def kayitli_ogrencileri_goster():
    if not cursor: return
    cursor.execute("SELECT ad, soyad, gun FROM ogrenciler")
    rows = cursor.fetchall()
    win = tk.Toplevel(root)
    win.title("Kayıtlı Öğrenciler")
    text = tk.Text(win)
    for ad, soyad, gun in rows:
        text.insert(tk.END, f"{ad} {soyad} - {gun}\n")
    text.pack()

def delete_student():
    if not cursor: return
    ogr_id = simpledialog.askinteger("Sil", "Öğrenci ID:")
    if not ogr_id: return
    cursor.execute("SELECT ad, soyad, foto_yolu FROM ogrenciler WHERE id=%s", (ogr_id,))
    row = cursor.fetchone()
    if not row: 
        messagebox.showinfo("Bilgi", "ID yok.")
        return
    ad, soyad, foto = row
    cursor.execute("DELETE FROM yoklama WHERE ogrenci_id=%s", (ogr_id,))
    cursor.execute("DELETE FROM ogrenciler WHERE id=%s", (ogr_id,))
    db.commit()
    if foto and os.path.exists(foto):
        os.remove(foto)
    messagebox.showinfo("Başarılı", f"{ad} {soyad} silindi.")
    model_egit_ve_kaydet()

# ====================== ANA ARAYÜZ ======================
def main():
    global root
    root = tk.Tk()
    root.title("Yüz Tanıma Yoklama Sistemi")
    root.geometry("420x400")
    root.resizable(False, False)
    style = {"width": 30, "height": 1, "pady": 6}

    tk.Button(root, text="Yeni Öğrenci Ekle", command=yeni_ogrenci_kaydet, **style).pack()
    tk.Button(root, text="Yoklama Al", command=yoklama_al, **style).pack()
    tk.Button(root, text="Yoklama Sayacı", command=yoklama_sayac_goster, **style).pack()
    tk.Button(root, text="Excel'e Aktar", command=yoklamayi_excele_aktar, **style).pack()
    tk.Button(root, text="Kayıtlı Öğrenciler", command=kayitli_ogrencileri_goster, **style).pack()
    tk.Button(root, text="Öğrenci Sil", command=delete_student, **style).pack()
    tk.Button(root, text="Çıkış", command=root.quit, **style).pack()

    def on_close():
        stop_camera()
        if db: db.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        model_yukle()
    except Exception as e:
        print(f"Model hatası: {e}")

    root.mainloop()

if __name__ == "__main__":
    main()
