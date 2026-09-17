# arc-mycobot

<p align="center">
  <a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html"><img src="https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg" alt="Isaac Sim 5.1.0"/></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img src="https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg" alt="Isaac Lab 2.3.2"/></a>
  <a href="https://docs.python.org/3/whatsnew/3.11.html"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"/></a>
  <a href="https://releases.ubuntu.com/"><img src="https://img.shields.io/badge/platform-linux--64-orange.svg" alt="Linux platform"/></a>
</p>

Elephant Robotics **myCobot 280 JetsonNano**와 adaptive gripper를 Isaac Lab에서
PPO로 학습시킨다. 엔드이펙터 **위치 추종**(reach)과 **큐브 집어 옮기기**(lift)
두 task가 들어 있다.

로봇은 벤더가 배포하는 `mycobot_ros2` description에서 가져온다. 그 파일은 있는
그대로는 import되지 않는다 — XML이 well-formed가 아니고, 모든 joint가 velocity
limit을 0으로 선언하며, inertia를 가진 link가 하나도 없고, gripper는 `<mimic>`
다섯 개로 엮인 클러스터다. 이 복구는 저장소의 실질적인 일부이고, 문서화된 한
곳에서만 일어난다 — **`assets/robots/mycobot_urdf.py`**. 벤더 checkout은 절대
수정하지 않는다.

## 두 개의 task

| task id | 하는 일 | gripper |
| --- | --- | --- |
| `Isaac-Reach-MyCobot280JN-v0` | 엔드이펙터 **위치 추종** | 열린 상태로 용접 |
| `Isaac-Lift-Cube-MyCobot280JN-v0` | **25 mm 큐브**를 집어 목표 지점으로 운반 | sliding 평행 조 |

팔과 URDF 복구, 기구학은 둘이 공유한다. 다른 것은 gripper에 무엇을 허용하느냐
하나뿐인데, 그 하나가 물리 설정의 거의 전부를 바꾼다. → [lift task](#lift-task)

## 범위 — reach

시뮬레이션 안에서의 위치 추종. 정책은 자기 joint 상태와 목표 **위치**를 보고
joint 위치 오프셋을 낸다. 파지도, 접촉도, 자세(orientation) 추종도, 실기도 없다.

gripper는 **열린 상태로 용접**되어 flange에 병합된다. 질량과 형상만 기여하고 그
외에는 아무것도 하지 않는다. 한 번도 닫지 않는 task에서 mimic joint 다섯 개를
살려 두는 것보다 이쪽이 나은 이유는 `GRIPPER_FIXED_AT`에 적혀 있다.

## 빠른 시작

```bash
# 1. 벤더 로봇 description을 sibling checkout으로
git clone --depth 1 https://github.com/elephantrobotics/mycobot_ros2.git ../mycobot_ros2
#    (이미 있다면 $MYCOBOT_ROS2_DIR 로 가리켜도 된다)

# 2. 환경 구성 — Isaac Sim 5.1과 Isaac Lab 2.3.2가 pip 의존성으로 들어온다
uv sync

# 3. Isaac Sim은 첫 실행에서 NVIDIA Omniverse 라이선스 동의를 묻는데, uv 아래에서는
#    그 프롬프트에 stdin이 없다. 한 번, 의도를 갖고 수락한다:
export OMNI_KIT_ACCEPT_EULA=YES

# 4. GPU 시간을 쓰기 전에 로봇과 task를 먼저 검사한다
uv run pytest                   # ~0.3 s, GPU 불필요: URDF 복구 + task 기하
uv run workspace_sweep          # ~10 s, GPU 불필요: 모든 목표가 실제로 도달 가능한가
uv run list_envs                # 등록된 task id 목록

#    pytest가 "43 passed"가 아니라 "5 passed, 38 skipped"로 끝나면 1단계가 안 된
#    것이다. 로봇을 건드리는 테스트는 벤더 checkout이 없으면 실패가 아니라
#    skip되므로, 초록색으로 보여도 통과가 아니다. workspace_sweep이 정확한
#    경로와 clone 명령을 알려 준다.

# 5. 학습 전에 환경이 도는 것을 눈으로 본다. --max_steps 를 주지 않으면
#    중단할 때까지 계속 돈다.
uv run zero_agent   --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200
uv run random_agent --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200

# 6. 학습
uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless

# 7. checkpoint 재생
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
| DOF | revolute 6개, `joint2_to_joint1` … `joint6output_to_joint6` |
| root link | `joint1` — 고정 베이스이자 목표가 표현되는 프레임 |
| 추종 body | `joint6_flange` — TCP가 아니라 공구 flange |
| 질량 | 0.968 kg = 팔 0.85 kg + gripper 0.118 kg. PhysX는 `joint6_flange`를 **0.138 kg**로 보고하는데, 용접된 gripper가 여기에 병합되기 때문이다(0.020 + 0.118). |
| 도달 범위 | flange 기준 수평 302 mm, `z` ∈ [−103, +447] mm |

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

복구된 파일은 `generated/`에 놓인다. **이 디렉터리는 git에 올리지 않고, 올려서도
안 된다.** `package://`를 풀면서 mesh 경로가 절대 경로로 바뀌기 때문에
(`/home/<사용자>/manipulation/mycobot_ros2/...`) 커밋해 봐야 다른 머신에서는
존재하지 않는 경로를 가리킨다. 저작물이 아니라 파생물이고, 필요할 때 알아서
만들어진다 — `build_mycobot_urdf()`가 디렉터리까지 생성하며, 벤더 파일이 출력물보다
새로우면 다시 만든다. 따라서 clone 직후 `generated/`가 없는 것이 정상이다.

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

4096 환경, seed 0, 1500 iteration을 483 s(8분)에:

| iteration | 큐브 들림 | 큐브 목표 도달 | 평균 보상 |
| --- | --- | --- | --- |
| 250 | 81.3% | 61.7% | 129 |
| 500 | 90.9% | 74.5% | 153 |
| 1000 | 93.0% | 81.0% | 163 |
| 1499 | **93.5%** | **84.2%** | 158 |

제어 스텝의 비율로 읽으면 된다. 큐브는 시간의 94%를 60 mm 위에서, 84%를 목표
근처에서 보낸다. 곡선이 꺾이지 않으므로 마지막 checkpoint를 그대로 쓸 수 있다.

## 구조

```
src/arc_mycobot/
├── assets/robots/
│   ├── mycobot_urdf.py        # 벤더 URDF 복구 + 모든 구조 상수
│   └── mycobot_280.py         # MYCOBOT_280_JN_CFG / _LIFT_CFG: spawn, actuator, gain
├── kinematics/urdf_fk.py      # 복구 URDF 기반 numpy FK; Isaac import 없음
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
└── scripts/
    ├── rsl_rl/{train,play}.py
    ├── environments/{zero_agent,random_agent,list_envs}.py
    └── tools/workspace_sweep.py
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

* **link별 질량과 관성.** Elephant Robotics는 총 질량(850 g), 가반하중(250 g),
  작업 반경을 공개하지만 link별 분배도 관성 텐서도 공개하지 않는다. 여기 질량은
  직렬 팔의 통상적인 테이퍼로 분배해 공개된 총합에 맞춘 것이고, 관성은 회전 반경
  30 mm의 등방 구 근사다. 자릿수와 총합은 맞지만 토크 수준의 작업에는 부족하다.
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
