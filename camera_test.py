import cv2

# Buka kamera
cap = cv2.VideoCapture(0)

# Periksa apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# Baca dan tampilkan frame
while True:
    # Baca frame
    ret, frame = cap.read()
    
    # Periksa apakah frame berhasil dibaca
    if not ret:
        print("Error: Tidak dapat membaca frame.")
        break
    
    # Tampilkan frame
    cv2.imshow('Camera Test', frame)
    
    # Keluar jika 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
print("Test kamera selesai.")