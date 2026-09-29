# arc-mycobot

<p align="center">
  <a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html"><img src="https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg" alt="Isaac Sim 5.1.0"/></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img src="https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg" alt="Isaac Lab 2.3.2"/></a>
  <a href="https://docs.python.org/3/whatsnew/3.11.html"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"/></a>
  <a href="https://releases.ubuntu.com/"><img src="https://img.shields.io/badge/platform-linux--64-orange.svg" alt="Linux platform"/></a>
</p>

Elephant Robotics **myCobot 280 JetsonNano**와 adaptive gripper를 Isaac Lab에서
PPO로 학습시킨다. 엔드이펙터 **위치 추종**(reach), **큐브 집어 옮기기**(lift),
손목 카메라의 **표식 화면 중앙 정렬**(visual align) task가 들어 있다.

두 작업은 저장소에 포함한 팔·카메라·그리퍼 URDF를 읽는다. Reach에는 고정된
그리퍼, lift에는 두 손가락이 미끄러지는 평행 조 변형을 사용한다. 두 파일 모두
상대경로 mesh와 공통 +45° `tool_mount`를 포함하므로 실행에 벤더 checkout이나
`generated/`가 필요 없다. 벤더 URDF의 결함과 복구 방법은
`assets/robots/mycobot_urdf.py`에 남겨 두었다.

## 세 개의 task

| task id | 하는 일 | gripper |
| --- | --- | --- |
| `Isaac-Reach-MyCobot280JN-v0` | 엔드이펙터 **위치 추종** | 열린 상태로 용접 |
| `Isaac-Lift-Cube-MyCobot280JN-v0` | **25 mm 큐브**를 집어 목표 지점으로 운반 | sliding 평행 조 |
| `Isaac-Visual-Align-MyCobot280JN-v0` | RGB에서 검출한 빨간 표식 box를 화면 중앙에 정렬 | sliding 평행 조 |

팔과 카메라 장착 구조는 공유하고, lift의 그리퍼만 평행 조로 바꾼다.
→ [lift task](#lift-task)

## 범위 — reach

시뮬레이션 안에서의 위치 추종. 정책은 자기 joint 상태와 목표 **위치**를 보고
joint 위치 오프셋을 낸다. 파지도, 접촉도, 자세(orientation) 추종도, 실기도 없다.

gripper는 **열린 상태로 고정**된다. `camera_link`를 센서 장착 프레임으로 남기기
위해 fixed link를 병합하지 않는다.

## 빠른 시작

```bash
# 1. 환경 구성 — Isaac Sim 5.1과 Isaac Lab 2.3.2가 pip 의존성으로 들어온다
uv sync

# 2. Isaac Sim은 첫 실행에서 NVIDIA Omniverse 라이선스 동의를 묻는데, uv 아래에서는
#    그 프롬프트에 stdin이 없다. 한 번, 의도를 갖고 수락한다:
export OMNI_KIT_ACCEPT_EULA=YES

# 3. GPU 시간을 쓰기 전에 로봇과 task를 먼저 검사한다
uv run pytest                   # GPU 불필요: 패키지 자산 + task 기하
uv run workspace_sweep          # ~10 s, GPU 불필요: 모든 목표가 실제로 도달 가능한가
uv run list_envs                # 등록된 task id 목록

#    벤더 원본 복구 테스트만 벤더 checkout이 없으면 skip된다. 실행 경로는
#    저장소에 포함된 URDF를 검사하므로 checkout 없이도 검증된다.

# 4. 학습 전에 환경이 도는 것을 눈으로 본다. --max_steps 를 주지 않으면
#    중단할 때까지 계속 돈다.
uv run zero_agent   --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200
uv run random_agent --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200

# 5. 학습
uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless

# 6. checkpoint 재생
uv run play --task Isaac-Reach-MyCobot280JN-Play-v0 --num_envs 16
```

`uv run train`은 `logs/rsl_rl/reach_mycobot_280_jn/<timestamp>/`에 기록하며, 50
iteration마다 checkpoint를 저장하고 해석된 env·agent 설정을 `params/`에 덤프한다.
`uv run play`는 기본적으로 가장 최근 run을 집는다. 다른 것을 쓰려면 `--load_run`과
`--checkpoint`를 준다.

본 학습 전 smoke run:

```bash
uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless --num_envs 64 --max_iterations 5
```

볼 것은 **`Episode_Reward/end_effector_position_success`** — 목표에서
`SUCCESS_THRESHOLD`(20 mm) 이내에 머문 제어 스텝의 비율이다. 정책이 동작하는지를
말해 주는 단 하나의 숫자다.

### 문제 해결

**아무 출력 없이 몇 분간 멈춘 것처럼 보인다.** URDF 변환이 아니라 렌더러다. 이
로봇의 변환은 약 1.6 s면 끝난다. 이 머신에서 Isaac Sim의 첫 RTX 파이프라인
컴파일은 16분을 넘겼고, `SimulationContext.step()`은 `--headless`에서도 기본적으로
렌더링한다. 직접 물리 전용 스크립트를 쓴다면 `sim.step(render=False)`로 스텝하라.
여기 있는 스크립트들은 이미 그렇게 한다.

## 로봇

| | |
| --- | --- |
| 출처 | `mycobot_ros2/mycobot_description/urdf/mycobot_280_jn/mycobot_280_jn_adaptive_gripper.urdf` |
| 실행 자산 | `src/arc_mycobot/assets/robots/urdf/`의 고정 그리퍼 / 평행 조 파일 |
| DOF | revolute 6개, `joint2_to_joint1` … `joint6output_to_joint6` |
| root link | `joint1` — 고정 베이스이자 목표가 표현되는 프레임 |
| 추종 body | `joint6_flange` — TCP가 아니라 공구 flange |
| 질량 | 패키지 URDF 명목 합계 1.200 kg = Jetson Nano 팔 1.030 kg + adaptive gripper 0.110 kg + Camera Flange 2.0 0.060 kg. 카메라 없는 생성 URDF는 1.140 kg. 링크별 분배·관성·실제 USB 카메라 무게는 미측정. |
| 도달 범위 | flange 기준 수평 302 mm, `z` ∈ [−103, +447] mm |

패키지 URDF는 센서 부착용 가상 프레임 4개에 각각 0.1 g의 수치상 inertial을
추가했다. 따라서 실제 PhysX 합계는 약 **1.2004 kg**이다. 이전에는 이 프레임이
질량 없이 import되어 각 1 kg의 기본값을 받았으므로, XML의 1.200 kg이 실제
시뮬레이션 질량을 나타내지 못했다.

이 질량 변경 이전에 학습한 checkpoint의 동역학은 현재 URDF와 다르다. 시뮬레이션
평가를 다시 수행한 뒤 재학습 필요성을 판단해야 한다.

벤더는 자기 **link**들을 `joint1` … `joint6`이라고 부른다. 헷갈리지만 그쪽 것이다.
이름을 바꾸면 이 저장소와 `mycobot_ros2`가 영구히 어긋난다. link는 `jointN`,
joint는 `jointN_to_jointM`이다.

### URDF 복구가 고치는 것

넷 다 가정이 아니라 checkout에 대고 확인한 것이다:

| 결함 | 고치지 않으면 | 조치 |
| --- | --- | --- |
| `lower = "-2.932"1 upper = …` | well-formed XML이 아니라 모든 파서가 89번 줄에서 멈춘다 | 파싱 전 텍스트 수준 복구 |
| 13개 joint 전부 `velocity="0"` | 움직일 수 없는 PhysX joint로 import된다 | 2.0944 rad/s (120 °/s, 스펙시트) |
| `<inertial>`이 아예 없고, 팔 6개 link에 `<collision>`도 없음 | 질량 0인 articulation | 명시적 inertial + `collision_from_visuals` |
| gripper의 `<mimic>` joint 5개 | 다섯 중 하나만 바인딩된다 (아래 참조) | 용접(reach) / sliding 평행 조로 교체(lift) |

여기에 `<?xml version="1.1"?>` 선언, 남아 있는 `<xacro:property>`, 그리고 urdfdom이
ROS 아래에서만 해석하는 `package://` mesh URI도 함께 처리한다.

선택적 `build_urdf` 명령으로 벤더 원본을 복구하면 결과는 저장소 안이 아니라
`$XDG_CACHE_HOME/arc-mycobot/urdf/`에 놓인다
(`XDG_CACHE_HOME`을 지정하지 않으면 `~/.cache/arc-mycobot/urdf/`).
`package://`를 풀면서 mesh 경로가 이 머신의 절대 경로로 바뀌므로 커밋 대상이
아니다. `build_mycobot_urdf()`는 벤더 URDF, 복구 코드, mesh 루트와 gripper
모드의 서명이 바뀌면 캐시를 다시 만든다. Reach/lift 실행은 이 캐시를 읽지 않는다.

### 카메라 + 그리퍼 결합 URDF

`src/arc_mycobot/assets/robots/urdf/`의 `mycobot_280_jn_camera_gripper.urdf`는
reach용 고정 그리퍼, `mycobot_280_jn_camera_gripper_parallel.urdf`는 lift용
슬라이딩 평행 조다. 두 파일 모두 팔, 카메라 플랜지와 `camera_link`를 가진다.
필요한 STL 30개는 `src/arc_mycobot/assets/robots/meshes/`에 있으며 mesh 참조는
`../meshes/` 상대경로다. `mycobot_280.py`의 두 `UrdfFileCfg`와 CPU FK도
이 패키지 파일을 직접 읽는다.

이 자산은 `mycobot_curobo`의 2026-09-28 결합 URDF
(SHA-256 `1a74f874b9a55612c9bdc81e32a08e9dc49c2c57c622ca2407922e7a92250b14`)
를 이 저장소에 복사한 것이다. J6 flange 기준 `tool_mount`의 +45° yaw는
실물의 관절각 0 자세를 정면에서 본 관찰이며, 카메라는 그리퍼에 대해 +90° yaw다.
`camera_link`는 렌즈 중심의 추정 프레임으로, 광축 보정과 hand-eye calibration은
아직 하지 않았다. STL은 Elephant Robotics `mycobot_description` mesh에서
변환했으며, 배포 조건은 `meshes/LICENSE`에 보존했다.

평행 조 변형은 기존 lift 모델의 편측 13.1 mm 이동과 두 box pad 충돌체를
보존하고, 고정 그리퍼 파일의 카메라 장착 구조를 공유한다. 다른 link의 충돌체는
없다. 예전 reach/lift 학습 수치는 카메라 없는 모델에서 얻은 값이므로 현재
모델의 성능으로 해석하면 안 된다. 특히 lift의 시작 자세와 큐브 배치는 새
장착각·카메라 두께에 맞춰 다시 검증해야 한다.

## reach task

`Isaac-Reach-MyCobot280JN-v0`, manager 기반 `ManagerBasedRLEnv`.

| | |
| --- | --- |
| 관측 | 21 = joint 위치 6 + joint 속도 6 + 목표 위치 3 + 직전 action 6 |
| action | 홈 자세 기준 joint 위치 오프셋 6개, scale 0.5 rad |
| 제어 | 120 Hz 물리 위의 60 Hz 정책 (`decimation=2`) |
| 에피소드 | 12 s, 3 s마다 목표 재샘플링 → 에피소드당 목표 4개 |
| 목표 박스 | `joint1` 기준 (0.16, 0, 0.20) m 중심의 160 × 240 × 160 mm |
| 보상 | −0.2·‖e‖ + 0.1·(1 − tanh(‖e‖/0.05)) + 0.05·[‖e‖ ≤ 0.02], 여기에 action-rate·joint-velocity 페널티 |
| 종료 | 시간 초과만 |
| 성공 | flange가 목표에서 20 mm 이내 |

**위치만 본다.** 상류(upstream)의 자세 추종 보상은 의도적으로 뺐다. 도달 범위
280 mm의 6-DOF 팔에서는 쉽게 도달하는 위치라도 손목 자세가 단 하나로 정해지는
경우가 있어, 자세 보상을 켜면 gradient의 대부분을 "위치는 가능하지만 자세는
불가능한" 목표에 쓰게 된다. 다시 켜는 데 필요한 세 가지 변경과, 그중 세 번째가
왜 선택 사항이 아닌지는 `RewardsCfg`에 적혀 있다.

**목표 박스는 추측이 아니라 유도된 것이다.** `uv run workspace_sweep`이 이를 다시
유도한다. Isaac Lab이 변환하는 것과 같은 복구 URDF를 읽고, 선언된 joint limit에
대해 bounded least-squares IK를 돌리며, 내부뿐 아니라 여덟 꼭짓점을 명시적으로
검사한다. 꼭짓점이 중요하다 — 이전 박스는 균일 샘플링한 목표 300개를 통과했지만
한 꼭짓점이 9.3 mm 모자랐다. 균일 샘플링은 3차원 박스의 모서리를 사실상 방문하지
않기 때문이다.

## reach 결과

> [!warning] 아래 수치는 카메라 플랜지를 추가하기 전 모델의 기록이다. 새 패키지
> 자산의 학습 성능이나 기존 checkpoint의 재현 성능으로 간주하지 않는다.

RTX 5090, 4096 환경, seed 0에서 측정. task가 실제로 어디서 수렴하는지 보려고
1500 iteration 전 구간(1004 s)을 돌렸다:

| iteration | 평균 추종 오차 | 20 mm 이내 제어 스텝 |
| --- | --- | --- |
| 150 | 5.0 mm | 92.4% |
| **400** | **2.9 mm** | **93.0%** |
| 750 | 6.8 mm | 88.0% |
| 1499 | 4.9 mm | 88.6% |

**iteration 400에서 정점을 찍고 그 뒤로 나빠진다.** `Mean action noise std`가 run
동안 1.0에서 0.04로 무너진다. 대략 iteration 450을 넘기면 정책은 사실상
결정론적이 되어 탐색을 멈추고 표류한다. 그래서 `max_iterations`는 처음 설정했던
1500이 아니라 **500**이다. 더 오래 돌리면 15분을 더 쓰고 정책은 조금 더 나빠진다.

알아 둘 결과가 둘 있다:

* **마지막 checkpoint가 최선이 아니다.** `uv run play`는 기본적으로 최신 것을
  읽는다. 다른 것을 쓰려면 `--checkpoint <model_400.pt의 절대 경로>`를 준다.
  `--checkpoint`는 run 상대 이름이 아니라 *파일 경로*를 받는다.
* 500 iteration이면 약 5분 30초 걸린다.

`Episode_Reward/end_effector_position_success`를 해당 항의 가중치 0.05로 나누면
목표에 머문 제어 스텝 비율이 그대로 읽힌다 — 0.0465는 93%다. 빗나가는 7%는 대부분
목표가 재샘플링된 직후, 팔이 아직 이동 중인 순간이다.

**중요했던 단 하나의 숫자.** `JOINT_ACTION_SCALE`은 "이 팔이 UR10의 3분의 1
크기니까"라는 근거로 상류의 0.5에서 0.15로 낮춰 시작했다. 그 근거가 틀렸다.
action은 *joint* 공간에 있고, joint 변위는 팔 크기에 따라 줄지 않는다. 줄어드는
것은 직교 좌표 거리뿐이다. 0.15에서는 목표 박스가 95 백분위에서 9.8σ의 action을
요구했고, 학습은 39 mm까지 좋아졌다가 탐색 노이즈가 감쇠하면서 56 mm로 *퇴행*했다.
0.5에서는 3 mm까지 단조롭게 수렴한다. 유도 과정은 이 상수의 docstring에 있다.

성공은 **종료 조건이 아니라 보상 항**이다. 목표가 에피소드 안에서 재샘플링되므로
task는 도착이 아니라 추종이고, 지배적인 항이 음의 거리 페널티인 task에서 시간
초과가 아닌 종료를 두면 에피소드를 일찍 끝내는 것 자체가 보상이 되어 버린다.

## lift task

`Isaac-Lift-Cube-MyCobot280JN-v0`. 정책은 joint 상태와 큐브 위치, 목표 위치를 보고
joint 위치 오프셋 6개와 이진 열기/닫기 1개를 낸다.

| | |
| --- | --- |
| 관측 | 27 = joint 위치 7 + joint 속도 7 + 큐브 위치 3 + 목표 위치 3 + 직전 action 7 |
| action | joint 오프셋 6 (scale 0.5) + 이진 gripper 1 |
| 제어 | 100 Hz 물리 위의 50 Hz 정책 |
| 에피소드 | 5 s, 5 s마다 목표 재샘플링 |
| 큐브 | 25 mm, 20 g, 마찰 1.5 / 1.2 |
| 조 개폐 | 열림 45.6 mm, 닫힘 19.4 mm (datasheet 파지 범위 20–45 mm) |
| 들림 판정 | 큐브 중심이 60 mm 위 (놓인 상태는 13.5 mm) |

### 그리퍼를 실제로 집게 만들기

최종 형상은 벤더의 회전 링크가 아니라 **sliding 평행 조**다. `gripper="parallel"`
빌드는 다음을 만든다:

* **prismatic finger 2개.** `PARALLEL_FINGER_JOINTS`가 편측 13.1 mm씩 미끄러지고,
  하나의 이진 action이 둘을 (부호를 뒤집어) 함께 명령한다. 조는 45.6 mm에서
  19.4 mm까지 닫히는데, 이는 Elephant Robotics가 공개한 20–45 mm 파지 범위에
  맞춘 값이다. gripper 모델에서 datasheet가 뒷받침하는 유일한 숫자이므로
  테스트로 고정해 두었다.
* **대칭 box pad 2개.** `gripper_left3`/`gripper_right3`의 fingertip mesh를
  6 × 26 × 22 mm 박스로 교체하고 link 원점에서 12.3 mm 안쪽으로 넣었다. 시각과
  충돌 형상이 **같은** 박스다. 둘이 다르면 보이지 않는 형상이 물체를 집게 되고,
  이전 버전이 큐브를 하우징에 짓이기는 것처럼 보였던 원인이 그것이었다.
* **접근축 스윕 0 mm.** 열림에서 닫힘까지 pad의 접근 방향 범위는 35.4–61.4 mm로
  불변이다. 파지점 `JAW_OFFSET_IN_GRIPPER_BASE`는 그 중앙인 48.4 mm이고, 이는
  gripper 몸체가 닿는 거리 바깥이다.
* **측면 파지.** pad는 조 축을 기준으로 z ∈ [−11, +11] mm에 대칭으로 놓인다.
  접근은 수평이고 pad는 수직 평면에 선다.
* **로봇 전체 collider 2개.** 이 pad 둘이 전부다. `collision_from_visuals`를 끈
  결과인데, 켜 두면 팔이 장식용 visual mesh에서 뜬 convex hull을 달고 다니고,
  파지에 필요한 낮은 자세에서 그것들이 바닥을 누른다. 가만히 서 있기만 해도
  joint 오차가 **0.19 rad**(조 기준 약 30 mm, 큐브보다 크다) 쌓였다. 끄면
  **0.005 rad**로 떨어진다.

마지막 항목의 대가는 분명히 적어 둘 만하다: **팔의 link들이 지면과 서로를 통과할
수 있다.** task가 이를 보상하지도, 금지하지도 않는다.

### 여기까지 온 과정

처음 "동작하던" gripper는 허구였다 — 큐브가 집힌 게 아니라 하우징에 끼어 있었다.
그걸 정직하게 고치자 task는 네 번의 run 동안 *학습 불가능*해졌다. 각 run이 실제
결함을 하나씩 분리해 냈고, 발견된 순서가 이 기록의 핵심이다.

| 수정 | 증거 | 결과 |
| --- | --- | --- |
| PhysX mimic을 인접 joint로 재표적 | 벤더의 `<mimic>` 5개 중 4개가 빈 reference로 import; 오른쪽 손가락이 한계 0.7 rad를 넘어 1.10 rad까지 스윙 | 결합 비율 오차 0.6% 이내 |
| 파지점을 하우징 밖으로 이동 | fingertip 원점 중점이 body에서 16.8 mm인데 body는 13.9 mm까지 도달 — 32 mm 큐브가 13 mm 겹침 | 짓이기지 않고 집게 됨 |
| lift 전용 홈 자세 | reach 홈에서 측면 파지까지 **6.3σ**의 action이 필요 | 45 mm 고원에 머물던 조가 4–7 mm까지 내려옴 |
| `grasping_object` 보상 항 | 학습된 정책 probing: 2000개 샘플 *전부*에서 gripper 명령이 양수(열기). 한 번도 닫지 않았으므로 들어올림을 본 적이 없다 | 제어 스텝의 75%에서 조가 닫힘 |
| **sliding 손가락** | knuckle이 회전하는 동안 pad가 접근 방향으로 **15.2 mm 전진**해 큐브를 14–21 mm 밀어냄. pad를 어디에 붙여도 동일 | 0% → 94% |

`PhysxMimicJointAPI`는 실재하고, Isaac Lab의
`convert_mimic_joints_to_normal_joints=True`가 importer로 하여금 그것을 만들게
한다(이름이 오해를 부르므로 믿지 않고 확인했다). 다만 articulation mimic joint는
**parent-child로 인접한** joint만 묶는다. 벤더의 태그 다섯 개를 그대로 import하면
mimic prim 다섯 개가 생기지만 reference가 잡힌 것은 정확히 *하나*이고, 나머지
넷은 sibling branch의 `gripper_controller`를 가리키며 조용히 아무 일도 하지 않는다.

마지막 행이 이야기의 전부다. 회전 링크는 pad를 *평행하게* 유지하지만 *정지시키지는*
못하고, 그 호가 무는 대상을 끌고 간다. sliding 손가락에는 호가 없다. 크기 제한도
함께 사라졌다 — scripted grasp가 20, 22, 25, 28, 32, 36, 40 mm 큐브를 모두 잡는다.
회전 조는 34–42 mm에서만 됐다.

### lift 결과

> [!warning] 아래 수치는 카메라 플랜지를 추가하기 전 평행 조 모델의 기록이다.
> 새 자산으로 파지 성공률을 재측정하기 전까지 현재 성능을 뜻하지 않는다.

4096 환경, seed 0, 1500 iteration을 483 s(8분)에:

| iteration | 큐브 들림 | 큐브 목표 도달 | 평균 보상 |
| --- | --- | --- | --- |
| 250 | 81.3% | 61.7% | 129 |
| 500 | 90.9% | 74.5% | 153 |
| 1000 | 93.0% | 81.0% | 163 |
| 1499 | **93.5%** | **84.2%** | 158 |

제어 스텝의 비율로 읽으면 된다. 큐브는 시간의 94%를 60 mm 위에서, 84%를 목표
근처에서 보낸다. 곡선이 꺾이지 않으므로 마지막 checkpoint를 그대로 쓸 수 있다.

## visual align task

`Isaac-Visual-Align-MyCobot280JN-v0`는 packaged URDF의 `camera_link`에
`TiledCamera`를 붙인다. 128×128 RGB에서 빨간 표식의 box를 검출하지만
정책에는 RGB나 depth를 넣지 않는다. Box 5값은 꼭짓점 5개가 아니라
정규화된 `(cx, cy, w, h, detected)`이다. 앞 네 값은 box 중심과 크기,
마지막 값은 검출 유효성이다. 관절 위치·속도 12개와 이전 action 5개를
합쳐 정책 입력은 22개 값이다.

기본 자세는 J6(`joint6output_to_joint6`)만 `-pi/4`이고 다른 팔 관절은
0이다. J6은 카메라 영상의 45° 기울기를 바로잡으며 에피소드 중에도
고정한다. 5개 arm action에는 J6이 들어가지 않는다. 40 mm 빨간 큐브는
수평 카메라 앞에 두고 영상 가로·세로에 대응하는 Y/Z 시작 위치를 각각
±50 mm 바꾼다. 충돌 없는 kinematic 표식이며 Y/Z에서 진폭 25 mm,
주파수 0.10/0.13 Hz로 천천히 움직인다. 카메라 far clip은 0.6 m다.

보상은 box 검출·중앙 정렬·중앙 도달과 작은 action 변화·관절 속도
벌점으로 구성된다. 표식의 월드 좌표는 정책이나 보상에 넣지 않는다.
중력은 미보정 서보의 처짐을 분리하기 위해 꺼 둔다.

```bash
uv run train --task Isaac-Visual-Align-MyCobot280JN-v0 --num_envs 64 --headless --enable_cameras --max_iterations 300
uv run visual_align_eval --num_envs 64 --steps 90 --seed 20260930
uv run visual_align_eval --checkpoint logs/rsl_rl/visual_align_mycobot_280_jn_moving_box/2026-09-29_19-21-02/model_299.pt --num_envs 64 --steps 90 --seed 20260930 --image /tmp/visual-align-moving.png
```

현재 자세와 보상에서 학습한 checkpoint는 로컬
`logs/rsl_rl/visual_align_mycobot_280_jn_moving_box/2026-09-29_19-21-02/model_299.pt`에
있다. 64개 환경, PPO 300 iteration이다. 학습에 쓰지 않은 두 seed에서
90 step(3 s) 평가 후 마지막 20 step을 집계했다. 정렬 기준은 box 중심의
정규화 거리 `<0.1`(128×128 영상에서 약 6.4 pixel)이다.

| 평가 seed | zero action 정렬률 | PPO 정렬률 | PPO 평균 중심 오차 | 큐브 평균 이동 경로 |
| --- | ---: | ---: | ---: | ---: |
| 20260930 | 8.0% | 100% | 0.0102 | 5.08 cm |
| 20260931 | 6.0% | 100% | 0.0099 | 5.08 cm |

두 평가에서 표식 검출률은 100%였다. 같은 환경의 연속 프레임을
집계한 값이라 독립 episode 성공률은 아니다. 옛 자세와 보상에서 학습한
`visual_align_mycobot_280_jn_moving_xy`와 고정 큐브 checkpoint는
현재 환경에 재사용하지 않는다.

현재 검출기는 빨간색 임계값이다. 실물의 사람 손은 Open Images V7에
사전 학습된 `yolov8n-oiv7.pt`의 `Human hand` 클래스(267)를 사용해
추가 학습 없이 먼저 검출할 수 있다. [Ultralytics의 모델·클래스 문서](https://docs.ultralytics.com/datasets/detect/open-images-v7).
`model.predict(frame, classes=[267])`의 `Results`는 기존
`vision/yolo_boxes.py`에서 box 5값으로 변환할 수 있다. 실제 손목 카메라의
검출 성능, 지연, 누락, 렌즈 보정과 관절 구동은 아직 검증하지 않았다.


## YOLO Human hand visual align task

`Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-v0`는 위 빨간 표식 과제와
별개다. `assets/targets/hand_photo.usda`의 사진 표적을 손목 `TiledCamera`로
렌더링하고, 사전 학습된 Open Images V7 `yolov8n-oiv7.pt`를 **재학습 없이**
사용한다. 추론 결과는 `classes=[267]`로 `Human hand`만 남긴다. 모델은
카메라 RGB 256×256을 320×320으로 키워 추론한다. PPO에는 RGB가 아닌
`(cx, cy, w, h, detected)` 5값과 관절 상태·이전 action만 들어간다.
검출·중앙 정렬 보상도 같은 YOLO box를 사용한다. 손목 J6은 `-pi/4`로
고정하고 나머지 다섯 관절을 제어한다. 사진 표적은 Y/Z에서 천천히
움직인다. YOLO 가중치는 처음 실행 때 XDG cache 아래에 받으며
`ARC_MYCOBOT_YOLO_WEIGHTS`로 경로를 지정할 수도 있다.

```bash
uv sync --extra yolo
uv run --extra yolo train --task Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-v0 --num_envs 32 --headless --enable_cameras --max_iterations 300 --seed 20260929
uv run --extra yolo visual_align_eval --task Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-v0 --checkpoint logs/rsl_rl/visual_align_mycobot_280_jn_yolo_hand/2026-09-29_19-54-30/model_299.pt --num_envs 32 --steps 90 --seed 20260930
```

32환경 PPO 300 iteration을 281초 학습했다. 최종 checkpoint는 위
로컬 경로에 있으며 SHA-256은
`24b92b031a4221d27b0372a84f80337adf23a04c095efc48a856127c88d363e2`다.
다음은 학습에 사용하지 않은 seed에서 90 step을 실행하고 마지막 20 step을
집계한 결과다. 중앙 정렬은 정규화 box 중심 오차 `<0.1`(256×256 영상에서
약 12.8 pixel)로 정의했다.

| 평가 seed | 영점 action 정렬률 | PPO 정렬률 | PPO 평균 중심 오차 | PPO 손 검출률 |
| --- | ---: | ---: | ---: | ---: |
| 20260930 | 7.7% | 100% | 0.0070 | 99.97% |
| 20260931 | 0.9% | 100% | 0.0076 | 100% |

각 환경의 연속 프레임은 독립 episode가 아니다. 단일 환경의 동기화된
시뮬레이션·손목 카메라 영상(30 FPS, 109프레임)에서 손 box 중심은
`(0.470, 0.153)`에서 `(-0.0038, -0.0003)`으로 이동했고 표적은 약
6.19 cm 움직였다. 영상에는 실제 YOLO box를 그렸다. 대상은 생성한
**평면 손 사진** 한 장이다. 손의 3D 형상, 다양한 사람·조명·가림,
실물 카메라의 검출 지연과 팔 구동은 아직 검증하지 않았다. 따라서
이 정책의 실물 손 추종 성공을 뜻하지 않는다.

### Gravity-on domain-randomized hand policy

`Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-v0`는 기존 정책과 같은
22개 관측값·5개 action을 쓰되 중력을 켠다. Isaac Lab에서 읽은 로봇의 기본
질량은 1.2004 kg이다. 팔의 움직이는 link·그리퍼·카메라 플랜지 질량은
환경별로 0.8–1.2배, 관절 servo stiffness·damping은 0.85–1.15배로
시작 시 무작위화한다. 전체 장면의 중력은 3–5초 간격으로
`-9.3`–`-10.3 m/s²`에서 바뀐다. 관절 관측에는 위치 ±0.01 rad,
속도 ±0.04 rad/s 오차를 넣는다. YOLO box에는 episode별 최대 중심
표준편차 0.04, 크기 표준편차 12%, 누락률 12%, 미검출 시 가짜 box 확률
1%를 넣는다. 보상은 노이즈를 더하기 전의 실제 YOLO box로 계산한다.
기존 `visible` 보상이 YOLO 대신 빨간 큐브 함수를 읽던 오류도 고쳤다.

```bash
uv run --extra yolo train --task Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-v0 --num_envs 32 --headless --enable_cameras --max_iterations 600 --seed 20260929
uv run --extra yolo visual_align_eval --task Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-Play-v0 --checkpoint /path/to/model.pt --num_envs 32 --steps 120 --seed 20260930
```

이 무작위화는 한 장의 평면 손 사진에서 검출과 제어 오차를 흉내 낸다.
다양한 실제 손, 조명, 가림 및 Jetson 검출 지연을 검증한 것은 아니다.

600 iteration 실행의 마지막 checkpoint는
`outputs/arc-mycobot-yolo-hand-policy.pt`에 복사했다. 크기 492,347 bytes,
SHA-256 `b38c11966cba9985add2e46c0d3877479b50206b9593d81f2fdf28adc18a7a56`다.
같은 무작위화 평가 환경에서 기존 YOLO 정책과 새 정책을 동일한 seed로
비교했다(32환경 × 100 step, 마지막 20 step, 중심 오차 `<0.1`).

| 평가 seed | 이전 정책 정렬률 | 새 정책 정렬률 | 이전/새 평균 중심 오차 |
| --- | ---: | ---: | ---: |
| 20260930 | 25.2% | 100% | 0.1906 / 0.0220 |
| 20260931 | 24.7% | 100% | 0.1921 / 0.0216 |

두 정책 모두 같은 시작 분포·표적 궤적과 명목 물성 범위에서 평가했다.
평가 중 환경별 총 질량 범위는 각각 1.148–1.247 kg, 1.121–1.272 kg이었다.
연속 frame을 집계한 값이므로 독립 episode의 성공 확률이 아니다.
카메라와 그리퍼 실물을 계량하거나 실기 성능을 확인한 결과도 아니다.

### Hand policy preview from a file or Hugging Face

The trained RSL-RL checkpoint is a separate file; it is not committed to Git.
`hand_policy_predict` loads the deterministic actor directly, detects only
Open Images `Human hand` class 267 with the frozen `yolov8n-oiv7.pt` detector,
and prints the five policy actions. A missing detection is reported as
`detected=0`. This command does not send commands to a physical arm.

```bash
uv sync --extra yolo
uv run --extra yolo hand_policy_predict \
  --checkpoint /path/to/arc-mycobot-yolo-hand-policy.pt \
  --image /path/to/camera-frame.jpg \
  --angles-deg 0 0 0 0 0 -45
```

After uploading that checkpoint to a Hugging Face **model** repository, replace
`--checkpoint` with `--hf-repo username/repository`. The default Hub filename is
`arc-mycobot-yolo-hand-policy.pt`; use `--hf-file` if you rename it. `--hf-revision`
can pin a revision. The Hub library uses its normal local cache and credentials.
For one frame from a local webcam, use `--camera-index 0` instead of `--image`.
Supply the measured angles with `--angles-deg`, and optionally measured joint
velocities with `--velocities-deg-s` and the last executed action with
`--previous-action`. Without the optional inputs the one-frame preview assumes
zero velocities and no previous action. The printed simulation targets are
computed with a 0.25 rad action scale and J6 fixed at -45 degrees; they have
not been validated as safe commands for the physical robot. The reusable
`HandAlignPolicy.predict` API accepts one uint8 HWC RGB frame and the same
joint state in radians.

## Jetson Nano sim-to-real hand policy

JetPack 4.6 / Python 3.8용 독립 실행 코드는 [`deploy/README.md`](deploy/README.md)에 있다.
Hugging Face의 checkpoint를 검증해 내려받고 USB 카메라·YOLO·관절 피드백으로
정책을 실행한다. 기본은 read-only preview이며 `--execute`에서만
`mycobot-control`의 제어 gate를 통해 제한된 관절 명령을 보낸다.

## 구조

```
src/arc_mycobot/
├── assets/robots/
│   ├── camera_gripper_assets.py  # 실행 URDF의 CPU-safe 경로
│   ├── urdf/                     # 고정 그리퍼 / 평행 조 버전
│   ├── meshes/                   # 상대경로 STL + 벤더 라이선스
│   ├── mycobot_urdf.py           # 벤더 복구 도구 + 구조 상수
│   └── mycobot_280.py            # 두 패키지 URDF의 spawn, actuator, gain
├── kinematics/urdf_fk.py      # 패키지 URDF 기반 numpy FK; Isaac import 없음
├── vision/{boxes,yolo_boxes}.py # RGB 표식 및 YOLO box의 공통 관측 형식
├── tasks/reach/
│   ├── reach_env_cfg.py       # 로봇 무관 베이스 (scene, MDP, 60 Hz)
│   ├── mdp/                   # 프레임워크 항 + 성공 지표 + 목표 관측
│   └── config/mycobot/
│       ├── geometry.py        # 목표 박스와 길이 스케일 — 일부러 Isaac 비의존
│       ├── joint_pos_env_cfg.py
│       └── agents/rsl_rl_ppo_cfg.py
├── tasks/lift/
│   ├── lift_env_cfg.py        # 로봇 무관 베이스 (큐브, 목표, 보상 사다리, 50 Hz)
│   ├── mdp/rewards.py         # grasping_object — 보상 사다리의 "닫기" 단
│   └── config/mycobot/
│       ├── geometry.py        # 큐브·스폰·목표·파지점 — 역시 Isaac 비의존
│       ├── joint_pos_env_cfg.py
│       └── agents/rsl_rl_ppo_cfg.py
├── tasks/visual_align/
│   ├── visual_align_env_cfg.py # RGB box 관측·중앙 정렬 보상
│   ├── mdp/                    # 카메라 box 관측·보상
│   └── config/mycobot/
│       ├── joint_pos_env_cfg.py # TiledCamera와 표식 큐브
│       └── agents/rsl_rl_ppo_cfg.py
└── scripts/
    ├── rsl_rl/{train,play}.py
    ├── environments/{zero_agent,random_agent,list_envs}.py
    └── tools/{workspace_sweep,visual_align_eval}.py
```

구조를 지탱하는 경계가 둘 있다:

* **gym 등록은 지연된다.** `config/mycobot/__init__.py`는 *문자열* entry point를
  등록하고 `agents/__init__.py`는 아무것도 import하지 않는다. 그래서
  `import arc_mycobot.tasks`는 밀리초면 끝나고 GPU가 필요 없다.
  `tests/test_task_registration.py`가 `isaaclab`이 `sys.modules`에 들어오지 않음을
  단언한다.
* **기하 상수는 Isaac Lab을 피한다.** `geometry.py`와 `mycobot_urdf.py`는
  `isaaclab`을 import하지 않는다. 덕분에 `workspace_sweep`과 테스트 전체가 float
  여섯 개를 읽자고 Omniverse Kit 부트스트랩 비용을 치르는 대신 CPU에서 10초 안에
  돈다.

## 검증하지 않은 것

코드에 `TODO(unverified)`로 표시해 두었다. 여기서 나오는 숫자를 믿기 전에 알아 둘
것들이다:

* **link별 질량과 관성.** Elephant Robotics의 [SKU 4010100018 사양](https://americas.shop.elephantrobotics.com/collections/all-robotic-products/products/mycobot-280-jetson-nano)은 Jetson Nano 팔 1.030 kg,
  [adaptive gripper 사양](https://docs.elephantrobotics.com/docs/mycobot_280_jn_en/4-SupportAndService/Accessories/AdaptiveGripper.html)은 0.110 kg,
  [Camera Flange 2.0 사양](https://www.elephantrobotics.com/en/mycobot-camera-flange-en/)은 0.060 kg이다.
  링크별 분배와 관성 텐서는 공개되지 않아, 기존 가동 링크 추정치를 유지하고 팔의 추가
  0.180 kg을 베이스에 배분했다. 관성은 회전 반경 30 mm의 등방 구 근사다.
  카메라 플랜지의 0.060 kg은 해당 제품을 사용한다는 가정이며 실제 USB 카메라·케이블의
  질량과 플랜지 중심 질량 위치는 측정하지 않았다. 토크 수준의 sim-to-real에는 부족하다.
* **joint 강성·감쇠·토크 한계.** 서보 게인은 공개되어 있지 않다. 여기 값들은
  측정한 것이 아니라 고른 것이다. 측정한 것은 그 *효과* 쪽이다 — 홈 자세를 유지할
  때 최악의 정상상태 중력 처짐이 어깨에서 32.8 mrad이고, 그만큼 flange가 순기구학이
  말하는 위치보다 8.8 mm 아래에 놓인다. 이 저장소의 FK와 시뮬레이션 사이의 불일치는
  그 8.8 mm가 전부다.
* **joint 속도 한계.** 120 °/s는 팔 전체에 대한 스펙시트의 단일 수치이고, 그것을
  여섯 joint 모두에 적용했다.
* **gripper 강성·감쇠.** 파지력만 datasheet의 150 g 기준에 맞췄다.
* **목표 미도달 16%의 원인 분해.** 놓친 것인지 접근에 실패한 것인지 구분하지 않았다.

시뮬레이션 전용 task에는 이 중 무엇도 걸림돌이 되지 않는다 — 정책은 주어진 동역학이
무엇이든 그에 맞춰 학습한다. 다만 sim-to-real 전이 전에는 전부 실측해야 한다.
