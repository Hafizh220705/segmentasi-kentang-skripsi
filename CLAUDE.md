# Aturan Proyek

## Konteks penelitian
- Segmentasi MULTILABEL 4 jenis cacat permukaan kentang: 0 Scab, 1 Rot, 2 Greening, 3 Mechanical Damage.
- Arsitektur: U-Net dengan encoder ResNet34 pralatih ImageNet. Output 4 kanal, sigmoid per kanal.
- Satu piksel boleh memiliki lebih dari satu label. Anotasi yang berhimpitan dihitung di kedua kanal. Jangan pernah mengubahnya menjadi mask satu kanal berisi indeks kelas.
- Dataset: Defect Dataset dari Diamant Potato Dataset (Tasnia, 2026), format label poligon YOLO-seg ternormalisasi, split resmi train 1281, val 176, test 267.

## Aturan data
- `dataset/raw/` bersifat READ-ONLY. Jangan mengubah, me-rename, menghapus, atau menimpa file di dalamnya.
- `roi_dataset` tidak dipakai untuk training maupun evaluasi.
- Semua hasil olahan disimpan di `dataset/processed/` atau `outputs/`.
- Citra dataset sudah Green-Boosted. Jangan menerapkan green boost lagi pada citra dataset. Green boost hanya untuk citra baru saat inferensi.

## Aturan eksperimen
- Pemilihan model, hyperparameter, threshold, dan checkpoint HANYA berdasarkan validation set. Test set tidak boleh dipakai untuk tuning.
- Setiap run training disimpan di `outputs/runs/<nama_run>/` beserta salinan konfigurasi, seed, log, checkpoint, dan metrik.
- Gunakan nama metrik yang spesifik: IoU, Dice, Precision, Recall. Jangan memakai istilah "akurasi" untuk kinerja segmentasi.
- Kesalahan data (label tidak valid, file tanpa pasangan) harus DICATAT, tidak boleh dibuang diam-diam.

## Aturan menulis kode (kode ini akan dipelajari Hafizh untuk sidang)
- Kerjakan satu modul per sesi sesuai permintaan. Jangan mengerjakan modul berikutnya tanpa diminta.
- Sebelum menulis kode, jelaskan rencana singkat: fungsi apa saja, input dan output, serta alasan desainnya. Tunggu persetujuan untuk perubahan besar.
- Setiap fungsi wajib memiliki docstring berbahasa Indonesia yang menjelaskan tujuan, parameter, nilai kembalian, dan bentuk array (misalnya `(4, H, W)`).
- Beri komentar pada baris yang tidak jelas atau rawan salah, misalnya urutan x dan y, konversi BGR dan RGB, atau perhitungan float sebelum di-clip.
- Utamakan kode yang mudah dibaca daripada kode yang ringkas atau "pintar". Hindari abstraksi yang tidak perlu.
- Setiap modul disertai uji di `tests/` yang bisa dijalankan dengan pytest.
- Setelah modul selesai, buat ringkasan di `docs/<nomor>_<nama_modul>.md` berisi: tujuan modul, penjelasan tiap fungsi, keputusan desain beserta alasannya, dan cara menguji.
- Jangan mengubah kode yang sudah ada tanpa menjelaskan apa yang berubah dan kenapa.
- Jangan menginstal paket baru tanpa persetujuan.
