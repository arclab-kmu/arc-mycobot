# myCobot 280 JN hand-centering deployment

이 폴더는 Isaac Lab 없이 Jetson Nano에서 손목 USB 카메라, 사전 학습 YOLO `Human hand`, PPO 정책, myCobot 280을 연결한다. Python 3.8 문법이다. 기본은 **preview**이며 카메라와 로봇 관절을 읽지만 이동 명령은 보내지 않는다. `--execute`를 넣어야만 이 폴더의 안전 검사를 거쳐 `pymycobot.send_angles`를 호출한다.

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

처음 실행은 사진 한 장으로 Hugging Face 다운로드, 체크섬, YOLO와 정책 로드를 확인한다. 기본 모델은 [`arclab-kmu/arc-mycobot-yolo-hand-policy`](https://huggingface.co/arclab-kmu/arc-mycobot-yolo-hand-policy)의 revision `8d28bc59e05888c0dc06705a7e5ea5b64d2efe75`에 고정한다. 파일 SHA-256은 `24b92b031a4221d27b0372a84f80337adf23a04c095efc48a856127c88d363e2`여야 한다. 첫 실행에는 HF checkpoint와 `yolov8n-oiv7.pt`를 받으므로 인터넷이 필요하다. 다운로드한 checkpoint를 쓸 때는 `--checkpoint /path/to/arc-mycobot-yolo-hand-policy.pt`, YOLO를 별도로 준비했다면 `--yolo-weights /path/to/yolov8n-oiv7.pt`를 준다.

```bash
python -m deploy.run --image src/arc_mycobot/assets/targets/hand_palm.jpg --device cuda:0
```

실제 카메라에서는 먼저 preview만 실행한다. 로봇 연결은 Jetson UART `/dev/ttyTHS1` @ 1 Mbaud, 카메라는 `/dev/video0`이 기본이다. 관절 읽기·손 검출·프레임 방향·`observation_age_s`를 확인한다.

```bash
python -m deploy.run --device cuda:0 --max-frames 50
```

실기 명령을 내릴 때는 팔을 **수동으로** J1–J5 ≈ 0°, J6 ≈ -45°의 충돌 없는 시작 자세에 놓고 실행한다. 코드는 자동으로 홈 자세로 이동하지 않는다. 처음에는 짧게 관찰한다.

```bash
python -m deploy.run --device cuda:0 --execute --max-frames 20
```

카메라 방향이 예상과 다르면 `--rotate 90|180|270`, `--flip-x`, `--flip-y`를 preview에서 확인한다. 카메라 프레임은 중심을 정사각형으로 잘라 학습 때의 정사각형 영상 형식에 맞춘다. 종료는 `Ctrl+C`다.

## Command boundary

- 관측은 손 box 5개, J1–J6 상대각·속도 각 6개, 이전 action 5개로 22개다. RGB는 YOLO에만 사용된다. 정책 출력은 J1–J5 **절대 관절 목표각**이다. J6 목표는 -45°로 고정하며 그리퍼 명령은 없다.
- `--execute` 시작 시 J1–J5는 각각 ±10°, J6은 -45° ±3°여야 한다. 정책 목표와 관절 피드백은 J1–J5 ±30° 안에 있어야 하고, 한 번의 명령은 실측 각도에서 관절당 최대 1°만 이동한다. 속도 명령은 SDK 값 10, 목표 갱신 주기는 최대 5 Hz다.
- 손 검출 실패, 비정상 로봇 상태, 오래된 관측(1초 초과), 관절 읽기 오류나 카메라 오류가 나면 추가 명령을 중단하고 `stop()`을 요청한다. 토크는 끄지 않는다. preview에서는 명령을 보내지 않는다. `deploy/arm.py`가 세션 동안 `/tmp/mycobot_lock`을 잡는다. 실행 모드에서는 `set_fresh_mode(1)`을 확인한 뒤 명령을 보낸다.
- 이 제한은 소프트웨어 방어선이다. 학습은 한 장의 평면 손 사진과 시뮬레이션에 국한되어 실제 손, 카메라 보정, 지연, 관절 응답, 작업 공간 충돌은 검증하지 않았다. 실기 첫 시험에는 팔의 주변 공간과 정지 수단을 확보한다.
