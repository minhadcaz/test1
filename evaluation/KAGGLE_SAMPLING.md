# BIRD Dev: so sanh sampling voi SLM-SQL-1.5B tren Kaggle

Script `evaluation/compare_sampling.py` la code **moi cua repo nay**, khong co trong
checkpoint upstream neu ban chi `git clone` ban cu. Push thay doi len fork cua ban,
hoac upload file `evaluation/compare_sampling.py` thanh Kaggle Dataset va copy file
vao `SLM-PRJ/evaluation/` sau khi clone.

## Notebook cells

Bat GPU va Internet trong Kaggle Notebook. Cac cell sau gia dinh BIRD Dev duoc gan
tai `/kaggle/input/datasets/jenishk/bird-dev/dev_20240627/` nhu trong anh.

```python
!git clone https://github.com/mairiiscodin/SLM-PRJ.git /kaggle/working/SLM-PRJ
%cd /kaggle/working/SLM-PRJ
# Neu clone upstream chua co compare_sampling.py, chay dong tiep theo sau khi
# upload script moi thanh Kaggle Dataset (thay ten dataset cho dung):
# !cp /kaggle/input/slm-prj-evaluator/compare_sampling.py evaluation/compare_sampling.py
!python -m pip install -r requirements.txt -r requirements-model.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` phai la `True` de dung `--device cuda`. Khong cai de
PyTorch CPU de wheel GPU Kaggle san co. Tai ca hai checkpoint vao dung thu muc
project (can Internet va dung luong cho ca hai model):

```python
!python scripts/download_model.py --model-id cycloneboy/SLM-SQL-1.5B
!python scripts/download_model.py --model-id cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct
```

Neu `bird_eval.json` da import trong session nay, khong can import lai. Neu chua:

```python
!python scripts/import_bird.py \
  --questions /kaggle/input/datasets/jenishk/bird-dev/dev_20240627/dev.json \
  --database-root /kaggle/input/datasets/jenishk/bird-dev/dev_20240627/dev_databases \
  --output data/benchmarks/bird_eval.json
```

Chay thu 10 cau, sau do chay tap day du bang **output khac**:

```python
!python -m evaluation.compare_sampling \
  --dataset data/benchmarks/bird_eval.json \
  --output results/bird-10 --limit 10 --device cuda --with-adaptive
```

```python
!python -m evaluation.compare_sampling \
  --dataset data/benchmarks/bird_eval.json \
  --output results/bird-full --device cuda --with-adaptive
```

Dung `Ctrl+C` de dung mem: script ghi checkpoint moi 4 samples, ghi
`summary.json`/`summary.csv` voi **mau so la so cau da hoan thanh tung metric**.
`average_latency_ms` tinh ca schema/prompt, khong tinh model load hay chay gold;
`average_total_generations` tinh cac ung vien va merge/correction neu duoc goi.
Chay lai **dung lenh, dung dataset va output** de tiep tuc. Neu runtime bi kill
ngang, checkpoint tren dia van giu buoc da hoan tat; dung:

```python
!python -m evaluation.compare_sampling --output results/bird-full --report-only
```

Lenh tren dung lai `summary.json` va `summary.csv` tu checkpoint, khong can load
model. Muon lay ket qua truc tiep trong notebook:

```python
import pandas as pd
display(pd.read_csv('results/bird-full/summary.csv'))
```

**Luu y:** checkpoint tren `/kaggle/working` chi con khi file cua session do
duoc Kaggle luu lai. Neu session bi huy/mat du lieu ma chua Save Version (hoac
chua copy checkpoint sang storage ben ngoai), khong co cach resume tu do.
Chi dung lai cung output khi input dataset/config/model giong nhau; script se bao
loi neu khac. `--limit` lay N cau dau, khong phai mau ngau nhien.

## Dinh nghia va gioi han

- `greedy@1`: mot SQL temperature 0, EX theo `evaluation.metrics.execution_match`.
- `self_consistency@k`: bo phieu theo ket qua execution tren k ung vien sinh voi
  temperature 0.8; tie-break theo ung vien xuat hien truoc; SQL loi khong co phieu.
- `pass@k`: co **it nhat mot** ung vien co ket qua dung, la oracle upper bound,
  khong phai chat luong bo chon SQL.
- `slm_sql@k`: neu co it nhat hai nhom execution khac nhau, dua mot SQL moi nhom
  dan dau (kem preview toi da 5 dong) vao merge model rieng, sinh 8 revision,
  bo phieu ket qua revision va fallback ve SC neu khong co revision hop le. Neu
  <2 nhom hop le, dung ket qua SC. Khong dung correction module cua repo.
- Cac moc k=4, 8, 16, 32, 64 dung chung mot luong samples; checkpoint moi 4
  samples, tung moc k, tung cau hoi; sample seed xac dinh boi vi tri cau va
  chunk de resume khong lap candidates. Moi k ghi `completed/total` va `accuracy`.
- `--with-adaptive` ghi them hang `adaptive_sc` trong cung `summary.csv`. Goi dung
  pipeline adaptive cua project, voi `max_candidates=32`, seed cua tung cau,
  ranking/correction theo `config.yaml`; ket qua va latency luu trong checkpoint.
  Runner hien tai khong checkpoint ben trong mot lan goi adaptive, nen neu bi
  kill dung luc goi nay se chay lai rieng cau do.

Day la ban **phong theo** CSC va evaluator local cua project; prompt generation,
preview merge, voting/tie-breaking va EX timeout/limit khong nhat thiet trung
voi implementation/evaluator chinh thuc. Khong goi cac ket qua nay la tai lap
64.84%/67.08% cua bai bao. Voi BIRD Dev 1,534 cau, cac gold SQL khong hoat dong
duoi gioi han executor se duoc ghi vao `errors` trong summary, khong am tham tinh
la sai. De so sanh voi adaptive sampling cua ban, chay tren cung dataset, model,
prompt va local evaluator; prompt merge va chi phi sinh 8 revision phai duoc
bao cao rieng. Tap 1,534 cau voi toi da 64+40 mau/cau co the vuot thoi luong
Kaggle Notebook; theo doi `summary.csv` theo tung luot chay va tiep tuc tu
checkpoint neu session da duoc luu.
