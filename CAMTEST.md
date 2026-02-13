# Camtest: бенчмарк камеры для Raspberry Pi

Этот документ описывает, как использовать `camtest.py` (бенчмарк CSI/V4L2/GStreamer) для проверки камеры и оценки производительности.

## Что делает скрипт

`camtest.py` прогоняет набор режимов по нескольким бэкендам и сохраняет отчёт:

- **Picamera2 (libcamera)** — прямой доступ к CSI-камере через `picamera2`.
- **OpenCV V4L2** — `VideoCapture` с `/dev/video*`.
- **OpenCV GStreamer (libcamerasrc)** — если OpenCV собран с GStreamer.

Выходные файлы:

- `./cam_benchmark_YYYYmmdd_HHMMSS/report.txt`
- `./cam_benchmark_YYYYmmdd_HHMMSS/samples/*.jpg`

## Требования

- **Python 3.8+**
- Зависимости Python:
  - `opencv-python`
  - `numpy`
  - `picamera2` (опционально, только для бэкенда Picamera2)
- Для GStreamer:
  - OpenCV, собранный с поддержкой GStreamer
  - Пакеты GStreamer и `libcamerasrc` (обычно идут через `libcamera`)

Установить зависимости из репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> На Raspberry Pi `picamera2` лучше ставить через `apt` (официальный пакет),
> чтобы работать с libcamera нативно.

## Быстрый старт

```bash
python3 camtest.py
```

По умолчанию скрипт:

- Пробует несколько разрешений (от 4056×3040 до 640×480).
- Пробует разные FPS (5–120).
- Замеряет FPS на каждом режиме и сохраняет пример кадра.

## Параметры запуска

```bash
python3 camtest.py --show
python3 camtest.py --frames 120
python3 camtest.py --max-v4l2 4
```

Параметры:

- `--show` — попытаться открыть окна предпросмотра (требуется графический дисплей).
- `--frames N` — сколько кадров брать для измерения FPS (по умолчанию 90).
- `--max-v4l2 N` — сколько индексов `/dev/video*` сканировать (по умолчанию 10).

## Как читать отчёт

Файл `report.txt` содержит:

- Информацию о системе (версия OpenCV, наличие GStreamer, `libcamera-hello --version`).
- Таблицу результатов по каждому бэкенду/устройству.
- Блок **BEST MODES** — лучший режим по измеренному FPS для каждого устройства.

Пример строки:

```
  1920x1080 @  30 fps | opened=YES configured=YES | measured=29.87 fps | frame=(1080, 1920, 3) uint8 | sample=samples/v4l2_dev0_1920x1080_30fps.jpg | err=-
```

Расшифровка:

- `opened/configured` — удалось ли открыть и настроить устройство.
- `measured` — фактический FPS по измерению.
- `frame` — форма массива кадра (высота, ширина, каналы) и тип данных.
- `sample` — путь к сохранённому JPEG.
- `err` — текст ошибки, если был сбой.

## Типовые сценарии

### Проверить, видит ли OpenCV камеру

```bash
python3 camtest.py --max-v4l2 2
```

Если в отчёте `OpenCV V4L2 devices: none`, камера не видна как `/dev/video*`.
В этом случае используйте Picamera2 или GStreamer (libcamerasrc).

### Быстро прогнать минимальный тест

Если нужно быстро проверить, что камера работает, можно временно ограничить сканирование:

```bash
python3 camtest.py --frames 30 --max-v4l2 1
```

### Использовать только Picamera2

Если `picamera2` установлена, в отчёте появится строка
`Picamera2 detected: YES`. Чтобы избежать лишних проверок V4L2/GStreamer,
можно ориентироваться на Picamera2-блок и игнорировать остальные.

## Частые проблемы и решения

- **Picamera2 не находится**: установите `picamera2` через `apt` и убедитесь,
  что камера включена через `raspi-config`.
- **Нет GStreamer**: пересоберите OpenCV с поддержкой GStreamer либо используйте V4L2.
- **Падение FPS на больших разрешениях**: это нормально для высоких разрешений;
  уменьшите разрешение или FPS в настройках скрипта.

## Где лежат результаты

Каждый запуск создаёт отдельную папку вида:

```
cam_benchmark_20250101_123000/
  report.txt
  samples/
    v4l2_dev0_1920x1080_30fps.jpg
    picamera2_cam0_2028x1520_30fps.jpg
```

Если нужно сохранить результаты — просто скопируйте папку целиком.
