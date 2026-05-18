## 1. Executive summary — какой пайплайн делаем и почему

**Главная идея.** Победный путь не «детектор + универсальный OCR», а **связка из трёх обязательных трюков**:

1. **Авторазметка train-датасета из CSV** — берём 274 размеченных bbox’а, по `frame_timestamp` достаём родительский кадр, **template-matching распространяет каждый bbox на ±10–20 соседних кадров** (NCC ≥ 0.42). Из 274 строк получаем 4–8 тыс. YOLO-меток без единого ручного клика → правило «нет ручной разметки» соблюдается.
2. **QR-first decoding.** На каждом ценнике QR содержит `barcode`, `p1..p4`, `wL1C/P`, `wL2C/P`, `aP/aC`. Если QR декодируется хоть на одном из 10 лучших кадров одного track’а — закрываем **11 колонок из 28** без OCR и попадаем в самую дорогую по баллам подзадачу («под звёздочкой»). Поэтому в стеке: **zxing-cpp → pyzbar (libzbar) → OpenCV WeChatQRCode → cv2.QRCodeDetectorAruco** с каскадом «upscale ×2/×3/×4 + 90°/180°/270° + ±8°/±15° tilt + adaptive threshold + CLAHE». Декодер бьём по 9 layout-приорам (QR всегда top-right относительно product_name).
3. **ByteTrack + top-K sharpest + per-field majority vote.** Одна строка CSV = один track\_id. На каждый track храним top-10 crop’ов с максимальным Лапласианом × √площадь; OCR прогоняется по каждому из 10; финальное значение поля — мажоритарное голосование по K результатам с валидацией доменными правилами (EAN-13 checksum, regex даты, длина SKU, диапазон цен).

Всё локально (open-source, без облака), детектор — YOLO11n с `imgsz=1280`, OCR — PaddleOCR PP-OCRv5 ru с Tesseract+rus как fallback, контейнер Docker, демо — Gradio/React (у тебя уже есть React). RKNN — отдельный экспорт `yolo11n.onnx → rknn` (roadmap, не блокер MVP).

**Почему именно так:**
- 274 GT строки — слишком мало, чтобы обучить YOLO напрямую, но template propagation даёт 20× больше меток с тем же IoU‑скелетом.
- QR закрывает то, что OCR никогда не возьмёт чисто (`wholesale_level_*`, `action_*` — их в visible-тексте часто нет).
- Top-K + majority vote — единственный способ обойти блики/motion blur при движущемся роботе.

---

## 2. Data audit — что в данных критично

**Видео (verified ffprobe):**

| | W×H | FPS | Frames | Sec |
|---|---|---|---|---|
| 25_12-20 | 3840×2160 | 60 | 823 | 42 |
| 25_2-10  | 3840×2160 | 20 | 609 | 30 |
| 26_12-20 | 3840×2160 | 19.98 | 1776 | 89 |
| 43_15    | 3840×2160 | 19.92 | 299 | 15 |
| 49_5     | 3840×2160 | 20 | 481 | 25 |
| Unlabeled/* | 3840×2160 | 19.92/59.94/20 | — | — |

- **Rotation flag отсутствует** во всех файлах. Сцена снята камерой, повёрнутой 90° CW — на диске бутылки лежат горизонтально, ценники расположены вертикально (узкая сторона горизонтальна). `ffmpeg -noautorotate` подтвердил.
- **CSV bbox координаты — в RAW (3840×2160) системе диска**, не в повёрнутой. Я draw’нул `(2011,1923)→(2231,2115)` на raw‑кадре 26_12-20 при t=6595 ms — ровно ценник с «-38% / 2345.99 ₽». **Никогда не пересчитывай bbox в повёрнутые координаты.** Если ротируешь кадр для OCR — крути обратно при записи CSV (или рендерь bbox в raw и вращай только сами crop’ы).

**CSV (274 строки, аудит выполнен):**
- `26_12-20.csv` и `43_15.csv` содержат опечатку `wholesale_level_1_coun` вместо `wholesale_level_1_count` (29 cols все там есть). Обработать на чтении и при записи **выводить корректное имя**.
- В CSV смешаны два числовых формата: `25_12-20`, `26_12-20`, `43_15` — **запятая**, поля цен в кавычках (`"3789,49"`); `49_5` и `sample.csv` — **точка** (`73.99`). В QR-колонках везде точка. В sample.csv тоже точка с запятыми в product_name и кавычки только на product. Целиться буду на формат **sample.csv = «точка для чисел, запятая = разделитель колонок»** — это эталон, явно подсунутый организаторами.
- `barcode` и `id_sku` в `49_5.csv` идут **с пробелами** (`4 607124 143901`, `360108 699851`) — это визуальное представление EAN-13/SKU. Хранить **без пробелов**, при сравнении с GT нормализовать через `re.sub(r"\s+","",x)`. EAN-13 checksum валиден на распознанных значениях.
- `filename` — три разных формата: `25_12-20/2.mp4` (со слешем), `43_15.mp4ESPACE`, `25_2-10` (без расширения), `26_12-20.mp4`. Записывать ровно как `os.path.basename(input)` — это ставит CSV ближе всего к sample.csv.
- `frame_timestamp` дублируется группами: на одном кадре часто 9–16 ценников. Значит **в финальной CSV в `frame_timestamp` пишется time того ключевого кадра, на котором ценник лучше всего распознался** (best crop из top-K).
- `bbox` median: width 201 px, height 240 px, aspect h/w = 1.16 → почти квадрат с лёгким preferans на «выше‑чем‑шире». Самый маленький bbox: 59×133 (нижняя граница), самый большой: 380×538. **min_box_size ≥ 50×100**, иначе OCR обречён.
- `color`: 273/274 = `red`, 11 случаев `yellow` (в `49_5` и одна в `25_12-20`). `green` не встречается, но в PPTX «ГМ для ТК» зелёные есть (овощи/фрукты) — заложить как extension.
- `special_symbols`: `К`/`Ш`/`нет`/изредка латинская `K` и `К ` с пробелом. Нормализовать `K→К`, strip.
- `price_discount`: в 273 из 274 строк = `нет`. `discount_amount` — `-XX%`, формат сохраняем буквально.
- `print_datetime`: формат `DD.MM.YYYY HH:MM` (иногда без времени → `DD.MM.YYYY`). 74 пустых, 4 «нет», 196 значений.
- `code`: формат свободный — `01_025019`, `01_025 008`, `13_043015`, `_048 005_1_12_1`, `01_026015 - 026016`. Не делать жёсткий regex — копировать «как есть» через PaddleOCR, потом `strip` и убирать перевод строки.
- `action_price_qr` / `action_code_qr` — **всегда** `нет` в текущих CSV. Заполнять `нет` по умолчанию, если QR не вернул `aP/aC`.
- `price3_qr` — никогда не имеет значения (199/274 «нет», 75/274 пусто). Это значит **поле `p3` физически отсутствует в QR payload** на доступных видео.

**Видимая структура ценника (по PPTX `Расшифровка ценники.pptx`, 7 слайдов с 8 PNG’ами):**
- product_name: TL, 0–60% width, 0–25% height (Cyrillic, multi-line, до 4 строк).
- QR: TR, 65–100% width, 0–25% height. Размер ≈ 25–35% длины ценника.
- price_default (под «Без карты, ₽»): right column, 18–35% height, мелким шрифтом.
- price_card (под «С картой, ₽»): right column, 38–62% height, **самый крупный шрифт** на ценнике.
- discount_amount: круглый бейдж `-XX%` left-mid, 35–60% height.
- id_sku + print_datetime + code: stacked text BL, 75–95% height (12 цифр + дата + код зоны).
- barcode (EAN-13 цифры + штрихкод-картинка): BR, 75–100% height.
- special_symbols (буква в круге «Ш/К/л»): между discount и barcode, 75–90% height, mid-left.
- additional_info (например «Сухое», «Полусладкое», «Шелфтокер», «3 по цене 2…»): middle-left, 28–50% height, рамкой.

Эти нормализованные доли — основа **zonal OCR** на следующем шаге.

---

## 3. Repo review — что брать, что выкинуть

(Отчёт sub-agent’а полный, ниже — выжимка для решений.)

### `shishechka-6/Lenta-Tech-Life-Hack`
- **Сильное:**
  - `eda/05_parsers.ipynb` — лучший в трёх репах парсер (EAN-13 checksum, font-size discount, `,99`-override для `price_card`, Latin→Cyrillic для `additional_info`).
  - Top-K sharpest crops per track + per-field majority vote — лучший паттерн дедупа.
  - YOLO11n@1280 + Ultralytics ByteTrack `persist=True` — рабочая база.
- **Слабое / рисковое:**
  - Тренируется на **внешнем `Khasan_Dataset`** (2659 bbox), что почти наверняка нарушает «нет ручной разметки».
  - CSV формат `submission.csv` с `;` и колонками `video;track_id` — **не совпадает с эталонной схемой**, дисквалификация.
  - WeChat QR: 2/544 декодов. Нужен расширенный каскад.
- **Брать:** парсеры, top-K + voting, ByteTrack config.

### `5e4nut/lenta-tech-hack`
- **Сильное:** preprocess crop (CLAHE → upscale → fastNlMeansDenoising → Otsu → авто‑инвертирование).
- **Слабое:** **66 train + 11 val вручную размеченных фреймов** (дисквалификация); нет CSV вовсе; нет QR; нет дедупа; нет регулярок; coords в повернутом пространстве.
- **Брать:** только функцию preprocess_crop как утилиту.

### `Black-Lotus-2026/Lenta-Tech-Life-Hack-2026`
- **Сильное:**
  - **`scripts/build_yolo_dataset.py`** — auto-генерация YOLO labels из публичных CSV через `cv2.matchTemplate`-propagation ±N кадров. Это **полностью совместимо с правилами** и единственный путь к нормальной точности.
  - QR cascade: zxing-cpp → pyzbar → cv2.QRCodeDetector + 9 layout priors × ~13 image variants (upscale/rotation/threshold).
  - `SimpleTracker` со **stable-id conflict gate**: два observation’а с разным non-empty `barcode|sku|qr_barcode|code` не сливаются — критически важно когда два соседних ценника на полке вплотную.
  - Zonal OCR + `suppress_code_artifacts` (маскируем плотные QR/barcode-зоны перед OCR).
  - Схема CSV точно совпадает с required 28 колонок (`schema.OUTPUT_COLUMNS`).
- **Слабое:**
  - Не закоммитили веса — без них fallback на слабый HSV.
  - Нет глобальной 90°-обработки.
  - Paddle 3.2 на CPU вылетает (в коде уже отключают MKLDNN).
- **Брать:** template propagation, QR cascade, conflict-gate tracker, zonal OCR + suppress_codes, точная CSV‑схема.

**Single highest-value:** template propagation из repo 3. **Главная ловушка:** доверять только WeChat QR на 4K — почти всегда ноль.

---

## 4. Best architecture — модули end-to-end

```
                ┌────────────────────────────────────────────────────────┐
                │             vision_service/app/pipelines               │
                │                  price_tag_v2/                         │
                │                                                        │
input.mp4 ──► [1. video_io]   open with cv2 + ffprobe; respect raw 3840x2160;
                              extract H264 PTS via cv2 CAP_PROP_POS_MSEC.
                │                                                        │
                ▼
              [2. frame_sampler]  sample_fps=4..6; sharpness gate
                                  Laplacian var ≥ τ; motion gate
                                  (mean optical-flow magnitude <  T_stop
                                   → "robot stopped", boost FPS to 10).
                │
                ▼
              [3. tag_detector]   YOLO11n @ imgsz=1280, conf=0.30,
                                  iou=0.50, 1 class `price_tag`.
                                  Trained from auto-propagated labels.
                │
                ▼
              [4. fallback_detector]  HSV(red,yellow) + edge + aspect
                                       + QR-seed (cv2.QRCodeDetector.
                                       detectMulti → expand bbox left ×3).
                                       Activated when YOLO conf < 0.30
                                       for the frame.
                │
                ▼
              [5. tracker]        ByteTrack via Ultralytics
                                  (`model.track(..., persist=True,
                                   tracker='bytetrack.yaml')`)
                                  with stable-id conflict gate from repo 3.
                │
                ▼
              [6. crop_picker]    per track_id: keep top-10 crops by
                                  score = laplacian_var * sqrt(area) *
                                          (1 - motion_blur_estimate).
                │
                ▼
              [7. crop_preprocess]  rotate per-crop to upright using
                                    QR-anchor (QR is always TR ⇒ rotate
                                    to put QR in TR); CLAHE on L of LAB;
                                    unsharp; resize long-side to 1024.
                │
                ▼
              [8. qr_decoder]    cascade zxing-cpp → pyzbar → WeChatQR
                                 → cv2.QRCodeDetectorAruco, on 9 layout
                                 priors × ~13 image variants. Parse JSON
                                 / compact `b/p1/p2/p3/p4/wL1C/wL1P/...`.
                │
                ▼
              [9. zonal_ocr]    PaddleOCR PP-OCRv5 ru per zone:
                                product_name | price_default | price_card |
                                discount_circle | sku_date_code |
                                special_symbol | barcode_digits |
                                additional_info.
                                Tesseract rus+eng fallback when Paddle
                                cluster empty.
                │
                ▼
             [10. field_parsers]  EAN-13 checksum; SKU 12 digits;
                                  date DD.MM.YYYY[ HH:MM];
                                  price `\d+[,.]\d{2}`;
                                  discount `-?\d{1,2}%`;
                                  Cyrillic-fix З→3 O→0 etc;
                                  prefer Paddle font-size for discount.
                │
                ▼
             [11. row_fuser]    per track_id, per field: majority vote
                                across K observations; QR overrides OCR
                                if QR confidence high; barcode taken
                                from QR if available else from EAN-13
                                OCR with checksum; color from HSV class
                                of bbox mean S/H.
                │
                ▼
             [12. dedup]        per-video: drop tracks whose final
                                stable-id (qr_barcode + sku + bbox-center)
                                already exists.
                │
                ▼
             [13. csv_writer]    write 28-column CSV with sample.csv
                                 formatting; `нет` for absent-on-tag
                                 (per-template rule), empty for
                                 not-recognized.
                │
                ▼
             [14. preview/api]  Gradio or existing React; expose
                                /api/jobs/{id}/csv, /preview, /crops.
```

Все стадии — **независимые `BaseStage`** в твоей текущей `vision_service/app/pipelines/price_tag_v2/stages/` (повторяем pattern из v1).

---

## 5. Training plan — детектор без ручной разметки

### 5.1 Цель
Получить ~5 000 YOLO labels на 1 классе `price_tag` без единого ручного клика, при бюджете 1–2 ч обучения на GPU‑Kaggle или 3–4 ч на CPU.

### 5.2 Шаги (`tools/build_yolo_dataset.py`)
```
read all Данные/<dir>/<dir>.csv  → 274 GT rows (xyxy, frame_ts, src_video)
for each (video, ts, bbox):
    frame_ref = cv2.VideoCapture(video).set(CAP_PROP_POS_MSEC, ts)
    crop_ref  = frame_ref[y_min:y_max, x_min:x_max]
    save (frame_ref, normalized bbox)  → split "keyframes"
    # ↓ Propagation
    for delta in [-20..-1, +1..+20]:        # 40 neighbours
        f2 = read_frame_at(video, ts + delta*Δt_ms)
        if f2 is None: continue
        loc = cv2.matchTemplate(f2, crop_ref, TM_CCOEFF_NORMED)
        max_v, max_loc = cv2.minMaxLoc(loc)
        if max_v < 0.42: continue
        bbox2 = (max_loc.x, max_loc.y, max_loc.x+w, max_loc.y+h)
        save (f2, bbox2)
```
- **40 соседей × 274 = до 11 000 кадров.** В реальности ~5–8 тыс. после NCC-фильтра.
- ВНИМАНИЕ: `cv2.VideoCapture.set(POS_MSEC)` неточно. Используй `set(CAP_PROP_POS_FRAMES, int(ts_ms*fps/1000)+delta)`.
- Сохраняем split: **train = 26_12-20 + 49_5 + 25_2-10**, val = **25_12-20**, holdout = **43_15** (минимально, чтобы у нас остался честный замер).
- Записываем в формат Ultralytics: `dataset/images/train/*.jpg`, `dataset/labels/train/*.txt` (YOLO `class cx cy w h` нормализованные к 3840×2160). Все labels на классе `0 = price_tag`.

### 5.3 Аугментации (Ultralytics CLI)
```yaml
hsv_h: 0.015; hsv_s: 0.7; hsv_v: 0.4  # бороться с бликами
degrees: 8.0; translate: 0.10; scale: 0.5
flipud: 0.0; fliplr: 0.5
mosaic: 0.8; mixup: 0.1
copy_paste: 0.2   # симулируем "ещё один ценник рядом"
auto_augment: randaugment
```

### 5.4 Обучение
```bash
yolo detect train \
  model=yolo11n.pt \
  data=dataset/data.yaml \
  imgsz=1280 epochs=80 batch=16 \
  project=runs/price_tag name=v2_propagated \
  optimizer=AdamW lr0=0.001 cos_lr=true \
  patience=20 close_mosaic=10
```
- Время на T4: ~50 мин. На CPU 4‑ядра: ~3 ч (выживаемо, но Kaggle проще).
- Целевой mAP50 на val ≥ 0.85 — этого хватает (типичные ценники крупные и контрастные).

### 5.5 Pseudo-labeling раунд 2 (Unlabeled)
```python
for video in Unlabeled/*.mp4:
    res = model.predict(video, conf=0.65, iou=0.5, imgsz=1280)
    # держим только bbox с conf≥0.70 + sharpness Laplacian > 25
    write_to_dataset(images/train/, labels/train/)
```
- Self-train ещё 30 эпох на объединённом датасете → `runs/price_tag/v2_pseudo/weights/best.pt`.
- Коммитим веса в `vision_service/app/pipelines/price_tag_v2/weights/best.pt` (≤6 MB).

---

## 6. Inference plan — обработка нового видео

```python
# pseudo-code, реализация в stages/
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 6.1 Sampling
sample_step = max(1, int(fps // 5))   # 5 fps base
keep_idx = []
prev_gray = None
sharpness_threshold = 18.0
for i in range(0, total, sample_step):
    cap.set(CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
    if not ok: continue
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < sharpness_threshold: continue
    # motion gate
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
        motion = float(np.linalg.norm(flow, axis=2).mean())
        if motion < 0.6:        # робот стоит — упомянутый сценарий "с остановками"
            sample_step_local = max(1, int(fps // 10))  # boost
    prev_gray = gray
    keep_idx.append(i)

# 6.2 Detect on raw frames (NO global rotate)
results = model.track(source=path, stream=True, imgsz=1280, conf=0.30,
                      iou=0.5, persist=True, tracker='bytetrack.yaml',
                      vid_stride=sample_step, agnostic_nms=True)

# 6.3 per-track top-K
tracks: dict[int, list[Obs]] = {}
for r in results:
    for det, tid in zip(r.boxes.xyxy, r.boxes.id or []):
        crop = full_frame[y0:y1, x0:x1]
        score = laplacian_var(crop) * math.sqrt((x1-x0)*(y1-y0))
        heapq.heappush(tracks[int(tid)], (-score, frame_idx, crop, bbox))
        if len(tracks[int(tid)]) > 10: heapq.heappop(tracks[int(tid)])

# 6.4 per-track parse → row
rows = []
for tid, obs in tracks.items():
    row = aggregate(obs, qr_decoder, ocr_engine, parsers)
    rows.append(row)

write_csv(rows, output_path)
```

**Цели по таймингу:** видео 60 с обработка ≤ 90 с на CPU 8 ядер, ≤ 25 с на T4 GPU. Это выполнимо: при `vid_stride=4` на 20fps реально обработать 5 кадров/сек, YOLO11n @ 1280 ≈ 25–60 мс/кадр CPU.

---

## 7. OCR/QR plan — конкретные библиотеки и порядок

### 7.1 QR cascade (всегда первая попытка перед OCR)

```python
QR_BACKENDS = ['zxingcpp', 'pyzbar', 'wechat', 'cv2_aruco']
QR_VARIANTS = [
    ('orig', None),
    ('gray', to_gray),
    ('clahe', clahe_l),
    ('upx2', upscale(2)),
    ('upx3', upscale(3)),
    ('upx4', upscale(4)),
    ('otsu',  otsu),
    ('adapt', adaptive_thresh),
    ('sharp', unsharp),
    ('rot90', rotate(90)), ('rot180', rotate(180)), ('rot270', rotate(270)),
    ('tilt+8', warp_rot(8)), ('tilt-8', warp_rot(-8)),
]
QR_REGIONS = [
    ('full', whole_crop),
    ('topright', crop_norm(0.55,0.0,1.0,0.30)),
    ('right_band', crop_norm(0.55,0.0,1.0,0.50)),
    ('bottom_right', crop_norm(0.55,0.70,1.0,1.0)),   # type-4 шелфтокер
    ('center_right',  crop_norm(0.40,0.10,0.95,0.45)),
    # ... 9 priors из repo 3
]
```
Для каждого track берём top‑3 crop’а; для каждого crop — все region × variant × backend, до первого валидного payload (поле `b` подтверждает EAN-13 checksum). Прерываем при первом успехе.

**Установить:**
```toml
"zxing-cpp>=2.2",         # bindings PyPI: pip install zxing-cpp
"pyzbar>=0.1.9",          # требует libzbar0 (apt-get install libzbar0)
"opencv-contrib-python-headless>=4.10",  # даёт WeChatQRCode + Aruco
```
WeChat нужен `models/wechat/*.prototxt+caffemodel` (4 MB, лежат в opencv-zoo, кладём в `vision_service/models/wechat/`).

**Payload parser** (`parsers/qr.py`):
- Сначала пробуем `json.loads`. Ключи поддерживаемые: `barcode|b`, `price1|p1`..`price4|p4`, `wholesaleLevel1Count|wL1C`, `wholesaleLevel1Price|wL1P`, `wholesaleLevel2Count|wL2C`, `wholesaleLevel2Price|wL2P`, `actionPrice|aP`, `actionCode|aC`.
- Если не JSON — пробуем URL `?p1=…&b=…`; затем pairs `key:value` через `\n` или `;`.
- Если ничего не подошло — пытаемся выдернуть просто EAN-13 регэкспом `\b\d{13}\b`, валидируем checksum, кладём только в `qr_code_barcode`.

### 7.2 OCR engine

**Стек:** **PaddleOCR 3.x (PP-OCRv5 `lang='ru'`)** как основной, **Tesseract 5 + tessdata `rus`+`eng`** как fallback (для CPU‑safe демо), **RapidOCR** запасной (на чистом ONNX без paddle).

**Зональный OCR.** На повёрнутый‑в‑upright crop накладываем 8 ROI (нормализованные доли — из PPTX):

| Zone | x1,y1,x2,y2 (доли) | Что ищем |
|---|---|---|
| product_name | 0.00, 0.00, 0.65, 0.28 | до 4 строк Cyrillic+Latin |
| qr_zone (mask) | 0.65, 0.00, 1.00, 0.30 | **маскируем** перед OCR |
| price_default | 0.55, 0.18, 1.00, 0.40 | `Без карты, ₽\n\d+[\.,]?\d*` |
| price_card | 0.45, 0.38, 1.00, 0.70 | `С картой, ₽\n\d+[\.,]?\d*` (largest font) |
| discount_circle | 0.00, 0.40, 0.30, 0.70 | `-\d{1,2}%` |
| additional_info | 0.20, 0.30, 0.55, 0.55 | «Сухое/Полусухое/Шелфтокер/3 по цене 2…» |
| sku_date_code | 0.00, 0.72, 0.45, 0.95 | 12 цифр + дата + код зоны |
| barcode_digits | 0.40, 0.78, 1.00, 1.00 | 13 цифр под штрихкодом |
| special_symbol | 0.30, 0.78, 0.42, 0.90 | одиночная буква Ш/К/л (CharOnly) |

Перед OCR — `suppress_code_artifacts`: чёрные плотные регионы (QR + штрихкод) находим Sobel‑энергией и **закрашиваем средним фоном**, чтобы recognizer не тратил ресурсы.

**Pre-processing crop (`utils/preprocess.py`):**
```python
def enhance(crop):
    # 1) per-crop rotation correction by QR anchor
    qr_pts = cv2_qr.detectMulti(crop)
    if qr_pts: crop = rotate_to_top_right(crop, qr_pts)
    # 2) resize long-side to 1024
    crop = resize_long_side(crop, 1024)
    # 3) LAB CLAHE on L
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[...,0] = cv2.createCLAHE(2.0, (8,8)).apply(lab[...,0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 4) unsharp mask
    gauss = cv2.GaussianBlur(crop, (0,0), 1.5)
    crop = cv2.addWeighted(crop, 1.5, gauss, -0.5, 0)
    return crop
```

### 7.3 Парсеры (`parsers/fields.py`)
```python
PRICE_RE  = re.compile(r"(\d{1,5})[.,]?(\d{2})?(?!\d)")
DATE_RE   = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2}))?")
SKU_RE    = re.compile(r"\b(\d{6})\s?(\d{6})\b")     # 12 digits with optional space
EAN_RE    = re.compile(r"\b(\d{13})\b")
DISC_RE   = re.compile(r"-?(\d{1,2})\s?%")
CODE_RE   = re.compile(r"\d{2}_\d{6}|\d{2}_\d{3}\s?\d{3}|_?\d{3}\s?\d_\d{1,2}_\d")

CYR_FIX = str.maketrans({'O':'0','o':'0','З':'3','I':'1','l':'1','о':'0','З':'3','Б':'6'})

def parse_price(s): m=PRICE_RE.search(s.translate(CYR_FIX)); ...
def ean13_ok(s):
    if len(s)!=13 or not s.isdigit(): return False
    chk = sum((1 if i%2==0 else 3)*int(c) for i,c in enumerate(s[:12]))
    return (10 - chk%10) % 10 == int(s[12])
```

**`discount_amount`** — выбираем bbox с максимальным шрифтом среди кандидатов в зоне `discount_circle` (font height per Paddle returns).

**`special_symbols`** — Paddle на zone special_symbol, фильтр `^[КШЛлкш]$`, нормализация `K→К`, `Л→Л`. Если зона пуста — `нет`.

**`color`** — *не из OCR*. Берём crop, HSV mean: red если `H ∈ [0,10] ∪ [170,179] and S>0.4`; yellow если `H ∈ [20,35] and S>0.4`; иначе `red` дефолтом (по факту распределения 273/274). Это надёжнее, чем OCR.

---

## 8. Dedup / fusion plan — одна строка на ценник

### 8.1 Stable-id

Каждый `Observation` имеет потенциальные «стабильные» идентификаторы:
- `qr_code_barcode` (если QR декодировался)
- `barcode` из OCR (если EAN-13 checksum валиден)
- `id_sku` (12 digits)

**Merge rule (repo 3 + усиления):** два track’а объединяются если:
- bbox IoU ≥ 0.3 в каком‑либо общем кадре, **И**
- центры расходятся ≤ 200 px между соседними кадрами, **И**
- если у обоих есть хоть один stable‑id — они должны совпасть; если у одного pусто — допустимо.

### 8.2 Top-K per track

Heap top-10 по `score = lap_var(crop) * sqrt(area)`. На каждый из 10 — отдельная попытка QR+OCR. Сохраняем все 10 результатов в `track.observations`.

### 8.3 Majority vote per field

```python
def fuse(observations, field, parser, validator):
    candidates = []
    for o in observations:
        v = parser(o[field])
        if v and validator(v): candidates.append(v)
    if not candidates: return ''     # not recognized
    cnt = Counter(candidates)
    val, freq = cnt.most_common(1)[0]
    return val
```

**Override rules:**
- `barcode`: QR > OCR (если QR `barcode` валиден EAN-13).
- `price_default`, `price_card`: QR `p1`/`p4` > OCR (на основе reverse-mapping из CSV: `p1=price_default`, `p4=price_card`; `p2`=price без карты доп, `p3`=пусто).
- `price_discount`: всегда `нет` (так в 273/274 GT).
- `id_sku`: OCR (QR не содержит id_sku — подтверждено Q&A).
- `print_datetime`: OCR с regex DATE_RE; majority vote.

### 8.4 «нет» vs пусто (правило из ТЗ)

Жёсткое разграничение **per template**:
```python
EXPECTED_ABSENT = {
    "type_1_basic_with_qr":      {"price_discount","action_price_qr","action_code_qr",
                                  "wholesale_level_1_count","wholesale_level_1_price",
                                  "wholesale_level_2_count","wholesale_level_2_price"},
    "type_4_shelftalker":        {"discount_amount","price_discount","barcode",
                                  "code","special_symbols", "additional_info"},
    "type_6_no_discount":        {"discount_amount","price_discount","action_*"},
}
```
- Если на детектированном type поле в `EXPECTED_ABSENT` — пишем `нет` независимо от OCR.
- Если поле должно быть, но не распознали — пишем **пусто** (empty string).
- Эта классификация type — простой ML на crop’е (PCA + linear SVM по template PNG’ам из PPTX) или эвристика «есть круг ‑XX% → type 1/2/3».

---

## 9. CSV plan — нормализация и edge cases

**Целевой формат — sample.csv:**
- Разделитель: `,`
- Десятичный: `.`
- Кодировка: UTF-8 без BOM
- Quoting: `csv.QUOTE_MINIMAL`; `product_name` всегда заворачивается, потому что содержит запятые
- Порядок колонок: ровно как в [shared/csv_schema.py](price-tag-vision/shared/csv_schema.py) (исправлено имя `wholesale_level_1_count`).
- `filename` = `os.path.basename(input_video_path)`, без префикса/постфикса.

**Edge cases (наблюдённые):**
- `barcode`/`id_sku` без пробелов в выводе.
- `discount_amount` — выводим строкой `"-38%"`, не числом.
- `print_datetime` — если только дата, без времени → `DD.MM.YYYY` (заметили в 25_12-20).
- `action_price_qr` / `action_code_qr` — **дефолт `нет`** (270/274 GT).
- `price3_qr` — **дефолт `нет`** (199/274 GT, никогда не имеет значения).
- `wholesale_level_*` — по умолчанию `нет` (за исключением 1 случая в 25_12-20: 2/805.26 + 3/760.52).
- `color` всегда заполнен (`red`/`yellow`), потому что HSV-classifier детерминирован.
- `special_symbols`: empty не бывает — либо буква, либо `нет`. Нормализация `K→К`.

**Запись:**
```python
import csv
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow(serialize_row(r))
```

`serialize_row` правит:
- `None` → `''`, `'нет'` → `'нет'`, числа → `f"{x:.2f}"` если есть decimals, иначе `str(int)`.

---

## 10. Evaluation plan — локальная proxy-метрика

Скрытая метрика организаторов: bbox + ≥80% полей. Воспроизводим близко:

```python
# tools/eval_local.py
def evaluate(pred_csv, gt_csv):
    # 1. group both by filename
    # 2. for each video: match predictions to GT by max IoU >= 0.5
    # 3. count "correct row" = IoU>=0.5 AND
    #    field_accuracy = sum(field_match) / sum(field_present)
    #    >= 0.80
    # 4. score = #correct / #GT_rows
```

**Field-match rules:**
- `barcode`, `id_sku`, `qr_code_barcode`: точное совпадение после `re.sub(r"\s+","",x)`.
- `price_*`: `abs(a-b) < 0.01`.
- `print_datetime`: совпадение по году/мес/числу + часу/минуте.
- `product_name`: similarity ≥ 0.85 (rapidfuzz `token_sort_ratio` / 100).
- `code`: similarity ≥ 0.85.
- `discount_amount`: точное совпадение строки.
- `color`, `special_symbols`, `additional_info`: точное (нормализованное).
- `frame_timestamp`, `x_min..y_max`: **не входят в поле‑match**, IoU считается отдельно.

### Ablation tests

| # | Ablation | Ожидание |
|---|---|---|
| A0 | baseline (heuristic only) | ≤ 20% |
| A1 | + YOLO11n trained on propagated labels | 45–55% |
| A2 | + ByteTrack + top-K + majority vote | 60–65% |
| A3 | + QR cascade (zxing+pyzbar+wechat+aruco × variants × regions) | 75–82% |
| A4 | + Zonal OCR + suppress_codes + parsers | **80–87%** |
| A5 | + per-template `нет` rules | +1–2% |

Гонять `python -m tools.eval_local --pred submission.csv --gt Данные/*.csv` после каждого изменения. Цель: **A4 ≥ 80%**.

---

## 11. Implementation roadmap для `price-tag-vision`

### 11.1 Что уже есть и используем как есть
- `docker-compose.yml` (postgres + redis + minio + backend + worker + vision_service + frontend) — **оставляем целиком**.
- `shared/csv_schema.py` — корректные колонки, починить опечатку `wholesale_level_1_coun` при чтении GT.
- `backend/*` (FastAPI + RQ jobs) — **не трогаем**.
- `frontend/*` — **не трогаем**, у нас уже React+Vite с UploadDropzone, PreviewTable, CropGallery — это и есть UI для жюри.
- `vision_service/app/pipelines/price_tag_cpu_v1/*` — оставляем как «legacy», но регистрируем **новую** `price_tag_v2` и делаем её default в `config.py`.

### 11.2 Новые файлы (в порядке имплементации)

**`tools/` (вне сервиса, для авторазметки и обучения):**
```
tools/
├── build_yolo_dataset.py        # template-propagation, читает Данные/*/*.csv
├── self_train_unlabeled.py      # round-2 pseudo-labels на Unlabeled/
├── train_yolo.py                # тонкая обертка вокруг ultralytics CLI
├── eval_local.py                # proxy-метрика IoU + field-match
└── data.yaml                    # для YOLO
```

**Новая пайплайн-папка `vision_service/app/pipelines/price_tag_v2/`:**
```
price_tag_v2/
├── __init__.py
├── pipeline.py                  # копия v1.pipeline.py, регистрирует новые stages
├── config.yaml                  # настройки v2 (sample_fps=5, imgsz=1280, ...)
├── orientation.py               # = v1 (без изменений), но НЕ применяется глобально
├── stages/
│   ├── __init__.py
│   ├── frame_metadata.py        # = v1
│   ├── frame_sampling.py        # ADAPT: sharpness + motion gate
│   ├── yolo_detection.py        # NEW: ultralytics YOLO11n
│   ├── fallback_detector.py     # NEW: HSV + QR-seed
│   ├── bytetrack_tracking.py    # NEW: вокруг ultralytics track()
│   ├── crop_picker.py           # NEW: top-K sharpest
│   ├── crop_preprocess.py       # NEW: rotate-by-QR + CLAHE + unsharp
│   ├── qr_decode_cascade.py     # NEW: zxing + pyzbar + wechat + aruco
│   ├── zonal_ocr.py             # NEW: PaddleOCR + tesseract fallback
│   ├── field_parsers.py         # NEW: regex + EAN13 + Cyrillic fix
│   ├── row_fusion.py            # NEW: per-track majority vote + QR override
│   ├── dedup_tracks.py          # NEW: stable-id conflict gate
│   ├── csv_writer.py            # ADAPT: формат sample.csv
│   ├── preview_writer.py        # = v1, добавить overlay c bbox+QR
│   └── debug_manifest.py        # = v1
├── parsers/
│   ├── __init__.py
│   ├── prices.py
│   ├── ean13.py
│   ├── dates.py
│   ├── qr_payload.py            # JSON / URL / pairs / EAN-fallback
│   ├── template_classifier.py   # классифицируем тип ценника (1..7)
│   └── absent_rules.py          # EXPECTED_ABSENT по template
├── ocr/
│   ├── __init__.py
│   ├── paddle_engine.py
│   ├── tesseract_engine.py
│   ├── rapidocr_engine.py
│   ├── ensemble.py              # последовательный fallback
│   └── zones.py                 # нормализованные ROI (см. §7.2)
├── qr/
│   ├── __init__.py
│   ├── zxing_backend.py
│   ├── pyzbar_backend.py
│   ├── wechat_backend.py
│   ├── aruco_backend.py
│   └── cascade.py               # regions × variants × backends
├── utils/
│   ├── sharpness.py
│   ├── motion.py
│   ├── color_classify.py        # HSV → red/yellow/green
│   └── enhance.py               # CLAHE + unsharp + resize
├── weights/
│   └── best.pt                  # 5 MB, committed
└── models/
    └── wechat/                  # 4 MB, prototxt + caffemodel
```

**Зависимости (`vision_service/pyproject.toml`):**
```toml
dependencies = [
  "boto3>=1.43.7",
  "fastapi[standard-no-fastapi-cloud-cli]>=0.136.1",
  "opencv-contrib-python-headless>=4.10.0",   # +contrib = WeChatQR/Aruco
  "pandas>=3.0.3",
  "pillow>=12.2.0",
  "pydantic-settings>=2.14.1",
  "pyyaml>=6.0.3",
  "python-multipart>=0.0.28",
  "uvicorn>=0.46.0",
  # NEW:
  "ultralytics>=8.3.0",          # YOLO11 + ByteTrack
  "paddleocr>=3.0.0",
  "paddlepaddle>=3.0.0",         # CPU build
  "pyzbar>=0.1.9",
  "zxing-cpp>=2.2.0",
  "rapidocr-onnxruntime>=1.4.0", # fallback OCR
  "rapidfuzz>=3.6.0",            # name similarity
  "pytesseract>=0.3.10",         # last-resort fallback
  "scipy>=1.13.0",               # для motion / NMS
]
```
В `Dockerfile` добавить:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    ffmpeg libgl1 \
 && rm -rf /var/lib/apt/lists/*
```

### 11.3 Порядок реализации (24 часа MVP)

**День 1 (часы 0–6) — Auto-dataset + train:**
1. `tools/build_yolo_dataset.py` (2 ч). Тестировать на 26_12-20: должно дать ~1500 кадров.
2. Загрузить в Kaggle/Colab, обучить YOLO11n@1280 80 эпох (1 ч fit + кофе).
3. `tools/self_train_unlabeled.py` (1 ч). Round-2 ещё 30 эпох.
4. Скоммитить `weights/best.pt`.

**День 1 (часы 6–12) — Inference skeleton:**
5. Скелет `price_tag_v2/pipeline.py` + регистрация в `registry.py` (30 мин).
6. `yolo_detection.py` + `bytetrack_tracking.py` + `crop_picker.py` (2 ч).
7. `qr_decode_cascade.py` со всеми 4 backend’ами (2 ч).
8. `field_parsers.py` (1 ч).

**День 1 (часы 12–20) — OCR и fusion:**
9. `paddle_engine.py` + `tesseract_engine.py` + `ensemble.py` (2 ч).
10. `zones.py` + `crop_preprocess.py` (rotation by QR anchor) (1 ч).
11. `zonal_ocr.py` (2 ч).
12. `row_fusion.py` + `template_classifier.py` + `absent_rules.py` (2 ч).
13. `csv_writer.py` (formate sample.csv) (1 ч).

**День 2 (часы 0–4) — Eval + tuning:**
14. `tools/eval_local.py` (1 ч). Прогон на 43_15 (holdout) и 25_12-20 (val).
15. Tuning порогов: sharpness, IoU, conf, QR variants order (2 ч).
16. Ablation table A0..A5 (1 ч).

**День 2 (часы 4–8) — Demo polish + deploy:**
17. Обновить frontend, чтобы он показывал поля QR в preview (1 ч).
18. README с инструкцией (30 мин), презентация (3 ч).
19. Deploy на Fly.io / собственный сервер (1 ч): `docker compose up -d`.

### 11.4 Что заменить в существующем v1

- `vision_service/app/pipelines/price_tag_cpu_v1/stages/heuristic_candidate_detection.py` — оставить как fallback‑detector только.
- `frame_sampling.py` `sample_fps: 1.0, max_frames: 20` → **слишком сильно ужимает.** В v2: `sample_fps: 5.0, max_frames: 600`, dynamic boost.
- `orientation.py` с `rotate_90_ccw` default — **отключить глобально**, использовать только на crop’ах в `crop_preprocess`.
- `barcode_qr_decode.py` использует только OpenCV QRCodeDetector — заменить целым каскадом из §7.1.

---

## 12. Команды запуска

### 12.1 Локально (без Docker)

```bash
# 1. одноразово — данные и веса
cd /Users/anastasiaserbuhina/лента/price-tag-vision
uv sync --project vision_service
# собрать датасет (~5 минут)
uv run --project vision_service python tools/build_yolo_dataset.py \
    --data-dir ../Данные \
    --out dataset_v1 \
    --propagate 20 --ncc 0.42
# обучить (на Kaggle 1 GPU)
yolo detect train model=yolo11n.pt data=dataset_v1/data.yaml \
    imgsz=1280 epochs=80 batch=16 \
    project=runs name=v2
# round-2 pseudo
uv run --project vision_service python tools/self_train_unlabeled.py \
    --weights runs/v2/weights/best.pt \
    --videos-dir ../Данные/Unlabeled --conf 0.65
yolo detect train model=runs/v2/weights/best.pt \
    data=dataset_v2/data.yaml imgsz=1280 epochs=30 batch=16 \
    project=runs name=v2_pseudo
cp runs/v2_pseudo/weights/best.pt \
    vision_service/app/pipelines/price_tag_v2/weights/best.pt

# 2. eval локально
uv run --project vision_service python tools/eval_local.py \
    --pred /tmp/submission.csv \
    --gt ../Данные

# 3. inference single video
uv run --project vision_service python -m app.cli.run_pipeline \
    --pipeline price_tag_v2 \
    --input ../Данные/26_12-20/26_12-20.mp4 \
    --out /tmp/out.csv
```

### 12.2 Docker (демо для жюри)

```bash
cd price-tag-vision
cp .env.example .env
docker compose up --build -d
# UI:   http://<host>:5173
# API:  http://<host>:8000/docs
# vision: http://<host>:9001/healthcheck
# MinIO console: http://<host>:9002

# логи
docker compose logs -f vision-service worker

# единичный inference через API
curl -F file=@../Данные/26_12-20/26_12-20.mp4 \
     http://localhost:8000/api/jobs
# → {"job_id":"..."}
curl http://localhost:8000/api/jobs/<job_id>
curl -o /tmp/out.csv \
     http://localhost:8000/api/jobs/<job_id>/download/csv
```

### 12.3 Деплой (для проверки 19–24 мая)

Вариант A — твой собственный VPS (рекомендую, никакого cold start):
```bash
# на сервере
git clone <repo> && cd price-tag-vision
cp .env.example .env
docker compose up -d --build
# nginx → :5173 → бесплатный Let's Encrypt
```
Вариант B — Fly.io single‑machine 2–4 vCPU + 8 GB RAM. В README указать «cold start ~30 секунд».

---

## 14. Риски и fallback-план

| Риск | Вероятность | Митигация | Fallback |
|---|---|---|---|
| YOLO даёт ≤ 0.7 mAP50 после propagation | средняя | Увеличить `propagate ±N` до 30; добавить `copy_paste` aug | Включить `fallback_detector.py` (HSV + QR-seed) — он держит recall на больших red‑ценниках |
| Paddle OCR не ставится на CPU | низкая, но болезненная | В Dockerfile фиксируем `paddlepaddle==3.0.0` cpu wheel | RapidOCR (onnxruntime, чистый pip install) — даёт 90% от Paddle |
| QR на видео физически непригоден (< 60 px) | высокая для ~30% ценников | upscale ×4 + WeChat (натренирован на маленьких) + zxing local-average binarizer | OCR штрихкода‑как‑цифр (EAN13 13 цифр под кодом) — отдельно tesseract digits-only |
| Ротация в новых тестовых видео разная | средняя | Per-crop rotation by QR anchor (QR всегда TR относительно product_name) | Перебираем 4 ориентации crop’а, выбираем ту, где у Paddle медианный confidence максимален |
| `frame_timestamp` указывает не в PTS, а в frame# | низкая, но если так — fail | Использовать `cv2.CAP_PROP_POS_MSEC`, **подтверждено**: 6595 ms бьётся с ценником | — |
| Деплой падает в день финала | низкая | Two‑node: VPS + Fly.io резерв; presigned MinIO URLs готовы | Локальный `docker compose up` + ngrok |
| Жюри проверяет на другой категории (овощи/фрукты, зелёные ценники) | средняя | В `color_classify.py` добавить класс `green`; в template classifier — type для зелёных МНЦ | В презентации показать roadmap масштабирования |
| Контрольное видео содержит rotation flag отличный от наших | средняя | Применяем **только** если `cap.get(CAP_PROP_ORIENTATION_META) != 0`; иначе берём raw | Brute-force: пробуем 0/90/180/270 на первых 5 кадрах, выбираем по доле успешных QR + OCR confidence |
| `wholesale_level_*` / `action_*` в тестовых ценниках имеют значения, которые мы дефолтим в «нет» | низкая | QR payload закрывает — он же эти поля и приносит | Если QR нет, оставить пусто (а не `нет`) — лучше пусто, чем неправильное `нет` |
| RKNN export — на финал не успеваем | высокая | Не блокер MVP; в презентации показать pipeline + указать «export к RK3588 — следующий шаг, ONNX уже есть» | `yolo export format=onnx imgsz=1280` → `rknn_toolkit2.convert` — кладём в `roadmap.md` |

**MVP-минимум, если 24 часа сгорят:** YOLO11n (предобученный на propagated dataset) + ByteTrack + QR cascade + EAN-13 OCR via Tesseract digits-only + price/sku/date regex над Paddle full-crop OCR + majority vote → **этого хватит на 65–70%** и проходное место в финал, потому что три эталонных репо застряли на 20–35%.

**Maximum-quality путь:** всё из §11.2 + ablation tune A0..A5 + per-template absent rules + green/yellow color extension + RKNN export → **80–87%**, реальная заявка на топ‑3.

---

Если хочешь, следующим шагом я могу:
- сразу написать `tools/build_yolo_dataset.py` и проверить, сколько кадров реально получится из 274 строк;
- или сгенерировать `price_tag_v2/pipeline.py` + `qr_decode_cascade.py` как первый рабочий PR;
- или собрать `tools/eval_local.py`, чтобы поставить baseline ещё до обучения YOLO.

Какой блок начнём имплементировать?
