# BABY STEPS: Instruksi Langkah-demi-Langkah

> File ini ditulis untuk model AI yang kurang capable atau junior developer.
> SETIAP langkah hanya melakukan SATU hal. Copy-paste command-nya saja.
>
> ATURAN: Jangan lanjut ke step berikutnya sebelum step sekarang BERHASIL.
> Jika gagal, BERHENTI dan minta bantuan.

---

## CARA PAKAI FILE INI

1. Baca satu step
2. Jalankan command yang ada di blok ```bash```
3. Cek apakah output sesuai "TANDA BERHASIL"
4. Jika berhasil, lanjut ke step berikutnya
5. Jika gagal, baca bagian "JIKA GAGAL" lalu coba lagi
6. Jika masih gagal, BERHENTI dan tanya user

---

## STEP 1: Cek Lokasi

```bash
cd D:\documents\aviation-anomaly
```

**TANDA BERHASIL**: Tidak ada error.

---

## STEP 2: Cek Data Ada

```bash
python -c "from pathlib import Path; p1=Path('data/cvr_labeled.csv'); p2=Path('data/processed/cvr_transcripts.csv'); print('DATA OK' if p1.exists() or p2.exists() else 'DATA MISSING')"
```

**TANDA BERHASIL**: Output `DATA OK`

**JIKA GAGAL** (DATA MISSING): Data belum ada. Jalankan:
```bash
python -m src.data.load_data
python scripts/add_temporal_labels.py
```

---

## STEP 3: Cek Dependencies

```bash
python -c "import torch; import transformers; import sklearn; import rich; import tqdm; print('DEPS OK')"
```

**TANDA BERHASIL**: Output `DEPS OK`

**JIKA GAGAL**: Install yang kurang:
```bash
pip install torch transformers scikit-learn rich tqdm pandas numpy seaborn matplotlib pyyaml python-dotenv
```

---

## STEP 4: Install Dependencies Tambahan

```bash
pip install xgboost sentence-transformers
```

**TANDA BERHASIL**: Tidak ada error merah.

**JIKA GAGAL** (xgboost): Tidak apa-apa, step selanjutnya tetap bisa jalan (skip XGBoost models).

---

## STEP 5: Jalankan Traditional Baselines (Exp 010) ✅ DONE

Ini step paling penting dan paling cepat. TIDAK butuh GPU.

```bash
python experiments/010_traditional_baselines/run.py
```

**STATUS: ✅ SELESAI Dijalankan 2026-02-06**

**TANDA BERHASIL**:
- ✅ Output menampilkan tabel dengan 9 model
- ✅ File `outputs/experiments/010/results.json` ada
- ✅ Best model: TF-IDF + SVM (Acc: 76.0%, F1: 0.635)

**ESTIMASI WAKTU**: 10-30 menit

**JIKA GAGAL** ("FileNotFoundError: data/cvr_labeled.csv"):
Buka file `experiments/010_traditional_baselines/config.yaml`, cari baris:
```yaml
source: "data/cvr_labeled.csv"
```
Ganti jadi:
```yaml
source: "data/processed/cvr_transcripts.csv"
```
Lalu jalankan ulang.

**JIKA GAGAL** ("No module named 'src'"):
```bash
pip install -e .
```
Lalu jalankan ulang.

---

## STEP 6: Verifikasi Hasil Exp 010

```bash
python -c "
import json
with open('outputs/experiments/010/results.json') as f:
    r = json.load(f)
n = len(r['models'])
print(f'Jumlah model: {n}')
if n >= 7:
    print('STEP 6 OK')
else:
    print('STEP 6 KURANG - hanya', n, 'model')
"
```

**TANDA BERHASIL**: `STEP 6 OK` dan jumlah model >= 7

---

## STEP 7: Cek GPU (Opsional tapi Penting)

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'TIDAK ADA')"
```

**JIKA ADA GPU**: Lanjut ke Step 8
**JIKA TIDAK ADA GPU**: Skip ke Step 12 (LLM Annotation)

---

## STEP 8: Jalankan DeBERTa+LSTM (Exp 011)

BUTUH GPU.

```bash
python experiments/011_deberta_lstm/run.py
```

**TANDA BERHASIL**:
- Training berjalan beberapa epoch
- `outputs/experiments/011/results.json` ada
- Accuracy > 0.70

**ESTIMASI WAKTU**: 2-4 jam

**JIKA GAGAL** ("CUDA out of memory"):
Buka `experiments/011_deberta_lstm/config.yaml`, ubah:
```yaml
training:
  batch_size: 2  # kurangi dari 4
```
Lalu jalankan ulang.

---

## STEP 9: Jalankan SMOTE (Exp 006)

BUTUH GPU.

```bash
python experiments/006_smote_augmented/run.py
```

**TANDA BERHASIL**:
- Training berjalan
- `outputs/experiments/006/results.json` ada

**ESTIMASI WAKTU**: 2-4 jam

---

## STEP 10: Jalankan K-Fold CV (Exp 014)

BUTUH GPU. Ini yang PALING LAMA.

```bash
python experiments/014_kfold_evaluation/run.py
```

**TANDA BERHASIL**:
- 5 fold selesai dilatih
- `outputs/experiments/014/results.json` ada
- Ada summary dengan mean dan std

**ESTIMASI WAKTU**: 10-20 jam

**TIPS**: Bisa jalankan per fold untuk hemat waktu:
```bash
python experiments/014_kfold_evaluation/run.py --fold 0
python experiments/014_kfold_evaluation/run.py --fold 1
python experiments/014_kfold_evaluation/run.py --fold 2
python experiments/014_kfold_evaluation/run.py --fold 3
python experiments/014_kfold_evaluation/run.py --fold 4
```

---

## STEP 11: Jalankan Window Ablation (Exp 008)

BUTUH GPU.

```bash
python experiments/008_ablation_window_size/run.py
```

**TANDA BERHASIL**: Results.json ada dengan 4 window sizes (5, 10, 15, 20)

**ESTIMASI WAKTU**: 8-16 jam

---

## STEP 12: Cek API Key DeepSeek

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); k=os.getenv('DEEPSEEK_API_KEY',''); print('API KEY OK' if len(k)>10 else 'API KEY MISSING')"
```

**TANDA BERHASIL**: `API KEY OK`

**JIKA GAGAL**: Buka file `.env` di root project, tambahkan baris:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```
(Ganti dengan key asli dari https://platform.deepseek.com)

---

## STEP 13: Jalankan LLM Annotation

BUTUH API KEY. Tidak butuh GPU. BUTUH BIAYA ~$2-5.

```bash
python scripts/annotation/llm_annotate.py --provider deepseek --rate-limit 0.5
```

**TANDA BERHASIL**:
- Progress bar berjalan
- `data/annotated/cvr_annotated_deepseek.csv` ada
- File CSV punya kolom `llm_score` dan `llm_label`

**ESTIMASI WAKTU**: 1-3 jam

**JIKA TERPUTUS** (koneksi putus, rate limit, dll):
Cek berapa baris sudah diproses, lalu resume:
```bash
python -c "import pandas as pd; df=pd.read_csv('data/annotated/cvr_annotated_deepseek.csv'); print('Baris selesai:', df['llm_score'].notna().sum())"
```
Lalu:
```bash
python scripts/annotation/llm_annotate.py --provider deepseek --resume-from 5000
```
(Ganti 5000 dengan angka dari output di atas)

---

## STEP 14: Jalankan Few-Shot LLM (Exp 012)

BUTUH API KEY. Test dulu dengan sample kecil.

```bash
python experiments/012_llm_fewshot/run.py --max-samples 50
```

**TANDA BERHASIL**: Output menampilkan metrics (accuracy, F1)

Jika berhasil, run full:
```bash
python experiments/012_llm_fewshot/run.py
```

---

## STEP 15: Jalankan Labeling Comparison (Exp 009)

BUTUH GPU + Step 13 selesai.

```bash
python experiments/009_labeling_comparison/run.py
```

**TANDA BERHASIL**: Tabel perbandingan 3 strategi ditampilkan

---

## STEP 16: Statistical Testing

```bash
python scripts/analysis/statistical_testing.py
```

**TANDA BERHASIL**: `outputs/analysis/statistical_results.json` ada

---

## STEP 17: Attention Visualization

```bash
python scripts/analysis/attention_visualization.py
```

**TANDA BERHASIL**: File PNG ada di `outputs/analysis/attention_maps/`

---

## STEP 18: Error Analysis

```bash
python scripts/analysis/error_analysis.py --exp-id 010
```

**TANDA BERHASIL**: File JSON ada di `outputs/analysis/error_analysis/`

---

## SELESAI!

Jika sampai Step 18, semua kode sudah dijalankan. Selanjutnya:

1. Update `paper.tex` dengan hasil baru
2. Tambah referensi ke `references.bib` (target 40+)
3. Generate figures untuk paper
4. Internal review
5. Submit ke Safety Science

---

## QUICK REFERENCE: Yang Harus Di-Edit Jika Path Data Berbeda

Jika data BUKAN di `data/cvr_labeled.csv`, edit `source:` di file-file ini:
- `experiments/009_labeling_comparison/config.yaml`
- `experiments/010_traditional_baselines/config.yaml`
- `experiments/011_deberta_lstm/config.yaml`
- `experiments/012_llm_fewshot/config.yaml`
- `experiments/013_finetuned_llm/config.yaml`
- `experiments/014_kfold_evaluation/config.yaml`

Cari baris `source: "data/cvr_labeled.csv"` dan ganti dengan path yang benar.
