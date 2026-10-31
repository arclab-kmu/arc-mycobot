# myCobot 280 JN hand-centering deployment

이 폴더는 Isaac Lab 없이 Jetson Nano에서 손목 USB 카메라, YOLO 손 검출기, PPO 정책, myCobot 280을 연결한다. 기본 손 검출기는 Hugging Face [`Bingsu/adetailer`](https://huggingface.co/Bingsu/adetailer)의 공개 `hand_yolov8n.pt`(한 클래스 `hand`)이며, PPO 정책만 이 프로젝트에서 학습했다. Python 3.8 문법이다. 기본은 **preview**이며 카메라와 로봇 관절을 읽지만 이동 명령은 보내지 않는다. `--execute`를 넣어야만 이 폴더의 안전 검사를 거쳐 `pymycobot.send_angles`를 호출한다.

## Environment

- 확인한 기기 조건: JetPack 4.6, Python 3.8.10, Torch `1.13.0a0+git7c98e70`, `torch.cuda.is_available() == True`. 실제 Jetson에서 이 코드의 import·속도·장치 연결은 아직 확인하지 못했다.
- **Torch 1.13이 import되는 바로 그 Python 3.8 환경**을 사용한다. 새 `venv`를 만들기만 하면 기존 Torch가 자동으로 들어가지는 않는다. 시스템 site-packages에 Torch가 설치된 경우에는 `python3.8 -m venv --system-site-packages .venv-deploy`를 사용할 수 있다. 다른 venv 안에 Torch가 있다면 그 venv를 그대로 사용한다.
- 해당 환경에서 `python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'`로 JetPack에 맞는 Torch/Torchvision 조합을 먼저 확인한다. Ultralytics는 Python 3.8+와 PyTorch 1.8+를 지원하지만 Jetson의 Torch/Torchvision 빌드는 일반 PyPI wheel과 다를 수 있다. [Ultralytics 설치 안내](https://docs.ultralytics.com/quickstart), [Jetson 안내](https://docs.ultralytics.com/guides/nvidia-jetson).

**배포 실행에 사용할 같은 Python 환경**에서 설치한다. 별도 로봇 제어 저장소는 필요하지 않다:

```bash
cd arc-mycobot
python -m pip install -r deploy/requirements.txt
python -c 'import sys, torch, torchvision, cv2, ultralytics, pymycobot; print(sys.executable, pymycobot.__file__, torch.cuda.is_available())'
```

`arc-mycobot`의 루트 `uv sync`는 Isaac Sim까지 설치하므로 이 Jetson 배포에는 사용하지 않는다. `deploy/arm.py`가 UART 잠금과 관절 피드백·명령 제한을 수행한다. ROS 노드 등 `/dev/ttyTHS1`을 사용하는 다른 프로세스와 동시에 실행하지 않는다.

## Run order

처음 실행은 사진 한 장으로 YOLO와 정책 로드를 확인한다. 새 gravity-on domain-randomized checkpoint는 `outputs/arc-mycobot-yolo-hand-policy.pt`에 준비했다(492,347 bytes, SHA-256 `b38c11966cba9985add2e46c0d3877479b50206b9593d81f2fdf28adc18a7a56`). 로컬 시험은 `--checkpoint`를 사용한다. Hugging Face의 [`arclab-kmu/arc-mycobot-yolo-hand-policy`](https://huggingface.co/arclab-kmu/arc-mycobot-yolo-hand-policy)에서 같은 파일명으로 **교체 업로드한 뒤**에는 기본 `main` revision이 이 새 SHA-256과 일치해야 실행된다. 2026-09-30에 원격 `main`(commit `f4ab0b25`)이 이 SHA-256과 일치함을 확인했다. 손 검출기는 `--yolo-weights`가 없으면 `Bingsu/adetailer` commit `53cc19de382014514d9d4038601d261a7faa9b7b`의 `hand_yolov8n.pt`(6,237,883 bytes, SHA-256 `3991202eb69e9ddcb3b9ba80cdeb41e734ffaf844403d6c9f47d515cd88c6f29`)를 받아 해시를 확인한 뒤 연다. 처음 한 번은 Jetson에 인터넷이 필요하고 이후에는 Hugging Face cache를 쓴다. 다른 가중치를 쓰려면 `--yolo-weights /path/to/weights.pt` 또는 `ARC_MYCOBOT_YOLO_WEIGHTS`를 준다. 런타임은 클래스 이름이 `hand` 또는 `Human hand`인 클래스를 정확히 하나 가진 모델만 받는다.

```bash
python -m deploy.run --checkpoint outputs/arc-mycobot-yolo-hand-policy.pt --image src/arc_mycobot/assets/targets/hand_palm.jpg --device cuda:0
```

실제 카메라에서는 먼저 preview만 실행한다. 로봇 연결은 Jetson UART `/dev/ttyTHS1` @ 1 Mbaud, 카메라는 `/dev/video0`이 기본이며 USB 카메라는 V4L2 backend로 연다. 시작 시 YOLO/CUDA를 팔 연결 전에 한 번 예열한다. 관절 읽기·손 검출·프레임 방향·`observation_age_s`를 확인한다.

```bash
python -m deploy.run --device cuda:0 --max-frames 50
```

YOLO 화면을 함께 보려면 별도 GUI 없이 로컬 browser preview를 켠다. 정사각형으로 자른 실제 카메라 영상에 `Human hand` 검출 박스(초록색), 영상 중심(흰색), 실행 모드와 관측 지연을 표시한다. `--max-frames`를 생략하면 `Ctrl+C`까지 계속 보인다.

```bash
python3 -m deploy.run --device cuda:0 --web-preview-port 8765
```

Jetson에서 브라우저를 연다면 `http://127.0.0.1:8765`로 접속한다. SSH로 접속 중이라면 **작업 PC의 별도 터미널**에서 `ssh -L 8765:127.0.0.1:8765 er@JETSON_IP`를 실행하고 작업 PC 브라우저에서 같은 주소를 연다. 영상 서버는 Jetson의 loopback 주소에서만 듣는다. 시작 전에는 "Waiting for camera frame"이 보일 수 있다.

첫 출력의 `warmup_s`는 명령 전 모델 준비 시간이다. 각 프레임의 `camera_read_s`, `inference_s`, `observation_age_s`로 지연 구간을 확인한다. `observation_age_s`가 1초를 넘으면 `--execute`는 명령 없이 멈춘다. 특히 매 프레임의 `inference_s`가 1초에 가깝거나 더 길면 관측 유효기간을 늘려 우회하지 말고 read-only preview의 수치를 먼저 확인한다. SSH 환경의 GDK 표시 오류가 남더라도 영상 창은 이 명령에서 사용하지 않는다. 카메라 자체가 열리지 않으면 `/dev/video0`의 존재·권한과 다른 프로세스 점유를 확인한다.

실기 명령을 내릴 때는 팔을 **수동으로** J1–J5 ≈ 0°, J6 ≈ -45°의 충돌 없는 시작 자세에 놓고 실행한다. 코드는 자동으로 홈 자세로 이동하지 않는다. 처음에는 짧게 관찰한다.

```bash
python3 -m deploy.run --device cuda:0 --execute --web-preview-port 8765 --max-frames 50
```

이전 `--max-frames 5` preview에서 팔이 움직이지 않는 것은 정상이다. 실행 모드의 terminal JSON에서 `box` 마지막 값이 `1.0`, `sent_target_deg`가 숫자 목록인지 확인한다. `angles_deg`는 실제 관절 피드백이다. `observation_age_s`가 매 프레임 1초 미만이고 로봇 상태가 정상일 때만 `--execute`를 시험한다. 검출된 손이 없으면 즉시 `stop()`을 요청하고 `motion_state: "waiting_for_hand"`로 대기한다. 카메라와 YOLO 화면은 계속 갱신된다. 손이 연속 두 프레임 검출되면 `motion_state: "tracking"`으로 돌아가 명령을 재개한다. 다시 놓치면 또 정지한다. 실행 모드에서도 `sent_target_deg`와 `angles_deg`가 거의 같으면 눈에 띄는 이동이 없을 수 있다.

2026-09-30 실기 로그의 프레임 66–73은 손 미검출로 대기했고, 프레임 74–76은 box 중심이 크게 변했다. 그 구간에서 `inference_s`는 약 0.08초, `observation_age_s`는 약 0.09초였다. 이를 보고 box와 관절 속도 관측, 목표각에 완만한 필터를 추가했다. 두 손이 검출되면 직전 손 box의 중심에 가까운 후보를 고른다. 정책 원목표 `policy_target_deg`가 ±30°를 넘으면 `target_clipped: true`를 기록하고, 목표를 안전 범위 안으로 투영한 뒤 프레임당 J1–J5 최대 0.6°씩 명령한다. J6은 정확히 -45°로 유지한다. 미검출 시 정지하는 동작은 그대로다. `cycle_work_s`, `status_read_s`, `joint_read_s`도 기록하므로 실제 루프가 0.2초를 넘는지 볼 수 있다.

카메라 방향이 예상과 다르면 `--rotate 90|180|270`, `--flip-x`, `--flip-y`를 preview에서 확인한다. 카메라 프레임은 중심을 정사각형으로 잘라 학습 때의 정사각형 영상 형식에 맞춘다. 종료는 `Ctrl+C`다.

## Command boundary

- 관측은 손 box 5개, J1–J6 상대각·속도 각 6개, 이전 action 5개로 22개다. RGB는 YOLO에만 사용된다. 정책 출력은 J1–J5 **절대 관절 목표각**이다. J6 목표는 -45°로 고정하며 그리퍼 명령은 없다.
- `--execute` 시작 시 J1–J5는 각각 ±10°, J6은 -45° ±3°여야 한다. 정책 목표는 J1–J5 ±30° 안으로 제한하고, 측정값이 이 범위를 벗어나면 명령을 거부한다. 한 번의 J1–J5 명령은 실측 각도에서 관절당 최대 0.6°만 이동한다. J6의 실측값이 -45°에서 1° 넘게 벗어나면 한 번에 보정할 수 없어 명령을 거부한다. 속도 명령은 SDK 값 10, 목표 갱신 주기는 최대 5 Hz다.
- 손 검출 실패 시 `stop()`을 요청하고 손을 다시 찾을 때까지 명령을 보류한다. 비정상 로봇 상태, 오래된 관측(1초 초과), 관절 읽기 오류나 카메라 오류는 실행을 종료하고 `stop()`을 요청한다. 토크는 끄지 않는다. preview에서는 명령을 보내지 않는다. `deploy/arm.py`가 세션 동안 `/tmp/mycobot_lock`을 잡는다. 실행 모드에서는 `set_fresh_mode(1)`을 확인한 뒤 명령을 보낸다.
- 이 제한은 소프트웨어 방어선이다. 학습은 한 장의 평면 손 사진과 시뮬레이션에 국한되어 실제 손, 카메라 보정, 지연, 관절 응답, 작업 공간 충돌은 검증하지 않았다. 실기 첫 시험에는 팔의 주변 공간과 정지 수단을 확보한다.

## External camera calibration

Jetson에는 `USB 2.0 Camera` 두 대가 `/dev/video0`(USB 2.1)와 `/dev/video1`(USB 2.4)로 잡힌다. 2026-10-06 두 카메라 화면에서 Camera 0은 로봇 옆 테이블 위의 외부 카메라, Camera 1은 그리퍼가 화면 왼쪽에 보이는 손목 카메라였다. 장치 번호가 그 순서와 같다면 `deploy.run`의 기본 `--camera /dev/video0`은 외부 카메라다. 정책에는 손목 카메라를 줘야 하므로 preview에서 그리퍼가 왼쪽에 보이는 장치를 `--camera`로 지정한다. Camera 1의 그리퍼 위치는 학습 시뮬레이션 화면과 같아 `--rotate`가 필요 없어 보였다.

`deploy/calibrate_camera.py`는 외부 카메라를 Isaac Lab의 `Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-View-v0`에 맞추는 JSON을 만든다. 카메라만 읽고 로봇 직렬 포트는 열지 않는다. 내부 모서리 9×6(칸 10×7) 체커보드를 평평한 판에 붙이고 칸 크기를 자로 잰다.

1. 렌즈 사진 25장을 모은다. 보드를 화면 중앙·네 모서리·가장자리로 옮기고 ±30° 정도 기울이며 거리도 바꾼다. 모서리가 검출되고 보드가 움직였을 때만 1초 간격으로 저장한다. 작업 PC에서 `ssh -L 8766:127.0.0.1:8766 er@JETSON_IP` 후 `http://127.0.0.1:8766`으로 검출 상태를 본다.

```bash
python3 -m deploy.calibrate_camera capture --camera /dev/video0 --out calib/cam0 --board 9x6 --count 25 --web-preview-port 8766
python3 -m deploy.calibrate_camera intrinsics --images calib/cam0 --board 9x6 --square-mm 25.0 --out calib/cam0_intrinsics.json
```

`rms_px`가 0.5 px 안팎이면 충분하고 1 px를 넘으면 더 선명하고 다양한 자세로 다시 찍는다. 해상도는 카메라 기본값으로 저장되며 Isaac Lab도 그 크기로 렌더링한다.

2. 보드를 로봇 옆 테이블 위에 평평하게 놓는다. 9개 모서리 방향(긴 변)을 로봇 +x(J1 = 0일 때 팔의 전방)와 평행하게 맞추고, 외부 카메라에 보드 전체가 보이게 한다. J1 축(로봇 베이스 중심)에서 보드 무늬 중심까지 x(전방)·y(왼쪽)를 m 단위로 잰다. 보드 두께도 준다.

```bash
python3 -m deploy.calibrate_camera extrinsics --intrinsics calib/cam0_intrinsics.json --camera /dev/video0 \
  --board 9x6 --square-mm 25.0 --board-center-m 0.20 -0.15 --board-thickness-mm 3 --out calib/external_camera0.json
```

체커보드 무늬는 180° 돌려도 같아 OpenCV가 어느 끝부터 모서리를 세는지 알 수 없다. 그래서 가능한 네 축 배치 중 카메라가 테이블 위에 있고 로봇 기둥이 화면 안에 들어오는 하나만 고른다. 둘 이상이거나 하나도 없으면 측정값과 보드 방향을 확인하라는 오류를 낸다. 결과 옆의 `_check.png`에는 base frame 축(빨강 +x, 초록 +y, 파랑 +z, 각 10 cm)과 J1 기둥(노랑, 0.44 m)을 그린다. 노란 선이 로봇 기둥을 따라 올라가야 한다. `_board.png`는 계산에 쓴 원본 프레임이다.

3. `calib/external_camera0.json`을 작업 PC의 `src/arc_mycobot/assets/cameras/external_camera0.json`으로 교체하거나 `external_view --calibration`으로 준다. 같은 시점의 실물 프레임과 관절각(`mc.get_angles()`, deploy가 실행 중이 아닐 때)으로 겹쳐 확인한다.

```bash
uv run --extra yolo external_view --calibration calib/external_camera0.json \
  --overlay calib/external_camera0_board.png --joint_angles_deg J1 J2 J3 J4 J5 J6 --rotate 270
```

## Hand detector

기본 검출기는 [`Bingsu/adetailer`](https://huggingface.co/Bingsu/adetailer)의 `hand_yolov8n.pt`다. 이전 fine-tune 모델과 같은 YOLOv8n(3.0M parameters)이며 클래스는 `hand` 하나다. 모델 카드는 Apache-2.0이고, Ultralytics로 학습한 가중치다. 기본 confidence는 0.4다.

2026-09-30 개발 PC에서 공개 모델을 비교했다. COCO-Hand test에서 가장 큰 손의 짧은 변이 48 px 이상인 300장을 골랐다. 손이 화면의 35–80%를 차지하도록 정사각형으로 잘라 손목 카메라처럼 만든 600장을 썼다. 손 annotation과 겹치지 않는 영역 300장도 따로 잘랐다. 전처리는 배포와 같은 320 bilinear다. "검출"은 가장 높은 confidence box가 annotation과 IoU 0.5 이상인 비율이다. "오검출"은 손 없는 crop에서 threshold 이상 box가 나온 비율이다.

| 모델 | confidence | 검출 | 오검출 |
| --- | ---: | ---: | ---: |
| Open Images `yolov8n-oiv7.pt`(정책 학습 때 검출기) | 0.05 | 12.7% | 0.0% |
| 이전 fine-tune `arc-mycobot-yolo-hand-detector.pt` | 0.25 | 44.0% | 29.3% |
| HaGRIDv2 `YOLOv10n_hands.pt` | 0.25 | 61.7% | 5.3% |
| ADetailer `hand_yolov8n.pt` | 0.25 | 74.0% | 12.0% |
| ADetailer `hand_yolov8n.pt`(기본값) | 0.4 | 71.2% | 5.3% |

HaGRIDv2 모델은 손가락이 아래를 향한 11k Hands 사진에서 22.8%만 검출해 제외했다. COCO-Hand annotation에서 빠진 손이 있으면 오검출이 과대추정된다. ADetailer 학습 데이터와 COCO의 중복은 확인하지 못했다. 모두 Jetson 실제 카메라 수치가 아니다.

손을 자주 놓치면 **먼저 read-only preview에서** `--yolo-confidence 0.25`를 시험한다. `hand_confidence`와 영상의 `best` 값은 threshold 아래의 후보 점수도 표시한다. 실제 손이 없을 때 오검출이 생기는지도 같은 설정으로 확인한 뒤 실행에 적용한다. threshold를 낮추면 손 검출률과 오검출이 함께 늘 수 있다. `box`는 YOLO 원검출, `policy_box`는 정책에 넣는 완만히 변하는 box다.

정책은 Open Images 검출기의 box로 학습했다. 같은 손 사진 `hand_palm.jpg`에서 Open Images box는 손목까지 포함한 `(cx, cy, w, h) = (-0.06, 0.02, 0.76, 0.98)`이다. ADetailer box는 손만 잡은 `(-0.07, -0.18, 0.79, 0.80)`이다. 이 차이를 확인하려고 Isaac Sim의 gravity-on domain-randomized 평가(`Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-Play-v0`, 32환경 × 100 step, 마지막 20 step, 중심 오차 `<0.1`)에서 같은 정책을 돌렸다. 임시 스크립트로 관측 box의 검출기만 ADetailer(클래스 0, threshold 0.4)로 바꿨고, 정렬 점수도 그 검출기의 box로 계산했다.

| 평가 seed | Open Images 정렬률 / 평균 중심 오차 | ADetailer 정렬률 / 평균 중심 오차 |
| --- | ---: | ---: |
| 20260930 | 100% / 0.0220 | 100% / 0.0203 |
| 20260931 | 99.8% / 0.0216 | 100% / 0.0199 |

시뮬레이션 대상은 여전히 평면 손 사진 한 장이다. 실기의 5 Hz 명령 주기와 프레임당 0.6° 제한은 이 평가에 들어 있지 않다.

## Train a hand detector

아래는 이전 fine-tune 기록이다. 위 비교에서 검출률이 낮고 오검출이 많아 기본값에서 뺐다. 기존 `yolov8n-oiv7.pt`는 [Open Images V7의 Human hand 클래스 267](https://docs.ultralytics.com/datasets/detect/open-images-v7/)를 그대로 사용한다. 손바닥 이외의 방향을 다루기 위해 [Ultralytics Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)의 공개 train/val 이미지와 박스 라벨로 **한 클래스 검출기**를 fine-tune한다. 포즈 라벨의 첫 5개 값에서 box를 만들고 이미지 경계를 넘는 박스는 잘라낸다. 이것은 공개 데이터를 사용한 추가 학습이며 처음부터 데이터와 모델을 만든 것은 아니다. 데이터 사용 조건은 원본 페이지의 CC BY-NC-SA 4.0 및 구성 데이터 출처를 확인한다.

CUDA가 있는 개발 PC에서 실행한다. 다운로드 파일과 학습 산출물은 Git에 추가하지 않는다.

```bash
curl -L --fail -o /tmp/hand-keypoints.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip
unzip -q /tmp/hand-keypoints.zip -d /tmp/hand-keypoints
python -m deploy.train_hand_detector \
  --pose-dataset /tmp/hand-keypoints \
  --dataset-output /tmp/hand-detect-data \
  --base-weights ~/.cache/arc-mycobot/weights/yolov8n-oiv7.pt \
  --project /tmp/hand-detector-runs --device 0
```

학습 결과의 `hand-detect/weights/best.pt`를 Jetson으로 복사한 뒤 `--yolo-weights /path/to/best.pt`를 추가한다. 런타임은 Open Images 클래스 267(`Human hand`), 이 fine-tune 모델의 클래스 0(`Human hand`), 공개 `hand` 모델을 모두 받는다. 기본 confidence는 각각 0.05, 0.25, 0.4다. `--web-preview-port 8765`에서 실제 손바닥·손등·측면을 각각 확인하고 `inference_s`와 `observation_age_s`를 본 뒤 `--execute`를 사용한다. 공개 데이터 검증 수치는 실제 Jetson 카메라에서의 검출률을 보장하지 않는다.

2026-09-30 학습본은 로컬 `outputs/arc-mycobot-yolo-hand-detector.pt`(6,205,098 bytes, SHA-256 `c6fe5de0c9708ab145c468d7215a681d5a6a2c0cce7ffb549943bdab4dc1a9de`)이다. 기존 Open Images 가중치에서 15 epoch 추가 학습했다. 가장 높은 검증 결과의 checkpoint는 14 epoch로, 공개 val 7,992장·7,992 box에서 precision 0.978, recall 0.984, mAP@0.5 0.992, mAP@0.5:0.95 0.888이었다. 같은 val에서 16장마다 한 장씩 뽑은 500장에 대해 실제 배포 confidence(기존 0.05, 새 모델 0.25)와 box IoU 0.5 기준을 적용하면 기존 172/500, 새 모델 490/500이 annotation에 맞았다. 이 데이터는 손이 있는 사진만 포함하므로 무손 프레임의 오검출률은 측정하지 못했다. Hojin이 보내 준 손바닥 화면 한 장에서는 새 모델과 기존 정책의 오프라인 추론이 성공했다. 손등·측면의 **Jetson 실물 검증은 아직 없다**.

```bash
python3 -m deploy.run --device cuda:0 --web-preview-port 8765 \
  --yolo-weights /path/to/arc-mycobot-yolo-hand-detector.pt
python3 -m deploy.run --device cuda:0 --execute --web-preview-port 8765 \
  --yolo-weights /path/to/arc-mycobot-yolo-hand-detector.pt
```
