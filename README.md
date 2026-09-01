# xArm5 키보드 원격조작 모방학습

Ubuntu 22.04 + ROS 2 Humble에서 xArm5의 그리퍼 끝으로 물체를 끌어 측면 바구니에 넣는 졸업작품용 기준 구현이다. 사람 손이 카메라에 들어가지 않도록 키보드로 시연하고, 카메라 이미지·5개 관절 상태·실제로 안전 필터를 통과한 TCP 속도 명령을 에피소드 단위로 저장한다. 이후 이미지와 관절 상태에서 미래 Cartesian 속도 묶음을 예측하는 행동 복제 정책을 학습한다.

노트북1·로봇 제어용 노트북2·학습 서버로 나누어 사용할 때는 먼저 `HANDOFF_LAPTOP1.md`를 읽는다.

이 코드는 연구용 시작점이다. 충돌 회피나 작업 성공을 보장하지 않는다. 실제 구동 전 작업 공간을 비우고, xArm 비상정지 장치를 손에 둔 채 저속·`dry_run`부터 시험해야 한다.

## 전체 흐름

```text
키보드(Shift+WASD/RF) ──> teleop_twist ─┐
                                         ├─> safety_mux ─> MoveIt Servo ─> xArm5
학습 정책 ─────────────────> policy_twist ┘       │
                                                  └─> 실제 실행 명령 기록

손목 카메라 + 관절 상태 + TCP pose + 실행 명령 ─> episode_XXXXXX.h5
episode들 ─> 검증 ─> 행동 묶음 Transformer 학습 ─> best.pt ─> policy_node
```

`safety_mux`는 조작/정책 명령과 로봇 사이에서 다음을 독립적으로 적용한다.

- 오래된 명령은 0으로 바꾼다.
- XY 및 Z 속도를 제한한다.
- `link_base -> link_eef` TF가 없으면 정지한다.
- 설정한 직육면체 작업영역 밖으로 나가는 축의 속도를 차단한다.
- 사람이 Shift를 누르면 정책보다 키보드 조작을 우선한다.
- 정책은 기본 비활성, 최대 실행 시간이 지나면 자동 비활성화된다.

MoveIt Servo 자체의 특이점 감속·충돌 검사도 함께 사용한다. 이 프로젝트가 명령을 보내는 기본 토픽은 xArm 공식 Servo 설정의 `/servo_server/delta_twist_cmds`이다.

## 카메라 구성

코드는 손목 카메라 한 대부터 동작한다. 그러나 **학습 가능 여부**는 별개다. 한 장의 현재 손목 영상에서 물체와 바구니의 상대 위치가 계속 보이지 않으면 정책이 필요한 정보를 알 수 없는 부분관측 문제가 된다.

먼저 손목 카메라 한 대로 다음을 확인한다.

1. 끌기 시작부터 바구니 투입까지 물체가 거의 계속 보인다.
2. 바구니 방향을 알려주는 고정된 시각 표식 또는 바구니 일부가 영상에 보인다.
3. 카메라 마운트, 높이, 노출, 조명을 수집과 실행 사이에 바꾸지 않는다.

조건을 만족하지 못하면 저가 USB 카메라 한 대를 고정된 전체 시점으로 추가하는 것이 가장 확실하다. 이때 `camera_names`와 `camera_topics`를 같은 순서로 추가한다.

```yaml
camera_names: [wrist, overhead]
camera_topics: [/camera_wrist/color/image_raw, /camera_overhead/color/image_raw]
```

학습 체크포인트에는 카메라 이름과 순서가 저장되므로 추론 때도 반드시 동일해야 한다.

## 1. 설치

### xArm ROS 2와 MoveIt Servo

xArm의 공식 ROS 2 저장소를 Humble 작업공간에 설치하고 실제 로봇용 MoveIt Servo가 먼저 동작해야 한다. 공식 저장소와 사용법은 [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2), Servo 개념과 안전 기능은 [MoveIt Servo 문서](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)를 참고한다.

```bash
mkdir -p ~/xarm5_il_ws/src
cd ~/xarm5_il_ws/src
git clone -b humble https://github.com/xArm-Developer/xarm_ros2.git --recursive
```

이 저장소의 `src/xarm_il` 폴더를 `~/xarm5_il_ws/src/xarm_il`로 복사한 뒤 설치한다.

```bash
cd ~/xarm5_il_ws
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
sudo apt install python3-pip python3-venv python3-pynput python3-opencv python3-h5py
python3 -m pip install --user -r src/xarm_il/requirements-ml.txt
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source ~/xarm5_il_ws/install/setup.bash
```

`src/xarm_il/requirements-ml.txt`는 Ubuntu 22.04의 ROS/OpenCV 바이너리와 맞도록 NumPy를 1.x로 제한하며 pip용 OpenCV를 설치하지 않는다. `cv_bridge`가 사용하는 NumPy/OpenCV는 Ubuntu 패키지 `python3-numpy`, `python3-opencv`를 우선한다. GPU 학습을 할 경우 PyTorch 설치 명령은 사용 중인 CUDA 버전에 맞게 [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/)에서 선택한다.

카메라 드라이버는 별도로 실행하고 다음처럼 실제 토픽을 확인한다.

```bash
ros2 topic list | grep -E 'image_raw|joint_states'
ros2 topic hz /camera_wrist/color/image_raw
ros2 topic echo /joint_states --once
```

## 2. 실제 환경에 맞게 설정

주 설정 파일은 `src/xarm_il/config/xarm5_il.yaml`이다. 최소한 다음 항목을 확인한다.

- `camera_topics`: 실제 카메라 토픽
- `joint_names`: `/joint_states`의 xArm5 관절 이름과 순서
- `axis_signs`: W/D/R 키가 의도한 +X/+Y/+Z 방향인지
- `workspace_min`, `workspace_max`: `link_base` 기준 TCP 허용 범위(m)
- `dataset_root`: 데이터 저장 위치

TCP 위치는 아래 명령으로 확인하면서, 실제 책상·바구니·로봇 자체와 충돌하지 않는 내부 범위를 측정한다.

```bash
ros2 run tf2_ros tf2_echo link_base link_eef
```

설정값은 임의로 넓히지 않는다. 직육면체 제한은 보조 안전장치일 뿐 장애물 형상을 표현하지 못한다.

## 3. 연결 및 `dry_run` 점검

카메라를 먼저 실행한 다음 xArm Servo와 수집 노드를 함께 띄운다. 첫 실행은 반드시 `dry_run:=true`로 한다.

```bash
source ~/xarm5_il_ws/install/setup.bash
ros2 launch xarm_il system_collect.launch.py \
  robot_ip:=192.168.1.XXX dry_run:=true
```

공식 xArm Servo를 이미 다른 터미널에서 실행했다면 다음만 실행해도 된다.

```bash
ros2 launch xarm_moveit_servo xarm_moveit_servo_realmove.launch.py \
  robot_ip:=192.168.1.XXX dof:=5
ros2 launch xarm_il collect.launch.py dry_run:=true
```

이 프로젝트를 사용할 때 공식 예제의 `ros2 run xarm_moveit_servo xarm_keyboard_input`을 동시에 실행하지 않는다. 그 노드는 안전 mux를 거치지 않고 같은 Servo 입력 토픽에 직접 명령을 보낼 수 있다. xArm 공식 Servo launch에는 joystick 입력 컴포넌트도 포함되므로 게임패드/SpaceMouse를 분리하고 별도의 `/joy` 발행 노드를 실행하지 않는다.

다른 터미널에서 5초 프리플라이트 검사를 실행한다.

```bash
ros2 run xarm_il preflight_check -- \
  --camera-topic /camera_wrist/color/image_raw --duration 5
```

`"pass": true`이고 카메라 메시지, 관절 상태, TCP pose, 안전 상태가 모두 들어오는지 확인한다. `dry_run`에서는 키 입력과 `/xarm_il/teleop_twist`가 변해도 로봇 및 `/xarm_il/executed_twist`는 0이어야 한다.

수집기는 `dry_run` 중이거나 센서가 준비되지 않았으면 Space 입력을 거부한다. 실제 teleop 녹화 중 idle/policy 모드로 바뀌어도 그 에피소드를 자동 폐기한다.

## 4. 키보드 조작 확인

키 입력은 전역 키보드 리스너를 사용한다. Ubuntu Wayland에서 입력을 받지 못하면 로그인 화면에서 “Ubuntu on Xorg” 세션을 선택한다. SSH만 사용할 경우 이 노드는 키보드가 연결된 GUI 세션에서 실행해야 한다.

| 키 | 기능 |
|---|---|
| 왼쪽 Shift를 누른 상태 | 이동 허용 데드맨 스위치 |
| W / S | base +X / -X |
| D / A | base +Y / -Y |
| R / F | base +Z / -Z |
| Space | 새 에피소드 녹화 시작 |
| 오른쪽 방향키 | 성공 에피소드 저장 |
| 왼쪽 방향키 | 현재 에피소드 폐기 |
| F8 | 키보드 모드 및 정책에서 사람 인계 |
| F9 | 정책 모드 선택(정책 활성 서비스도 필요) |
| Backspace 또는 Esc | idle 정지 요청 |

Shift를 놓으면 키보드 명령은 즉시 0이 된다. 키가 반대 방향이면 코드가 아니라 YAML의 `axis_signs`만 바꾼다.

작업 공간을 완전히 비운 후 속도를 더 낮춰도 좋다. 방향, TF, 작업영역 차단을 확인한 뒤에만 실제 출력으로 전환한다.

```bash
ros2 launch xarm_il collect.launch.py dry_run:=false
```

## 5. 데이터 수집

한 에피소드의 권장 순서는 다음과 같다.

1. 로봇을 시작 위치에 두고 물체 위치를 정한다.
2. 손을 카메라와 작업영역 밖으로 뺀다.
3. Space를 누른다.
4. Shift를 누른 채 천천히 물체를 끌어 바구니에 넣는다.
5. 성공했으면 오른쪽 방향키, 실패·충돌·물체 유실이면 왼쪽 방향키를 누른다.
6. 시작 물체 위치, 접근 방향, 작은 조명 변화를 골고루 바꿔 반복한다.

실패한 시연은 기본적으로 저장하지 않는다. 처음에는 30~50개로 전체 파이프라인을 확인한 뒤, 실제 학습용으로 최소 100~300개의 일관된 성공 시연을 목표로 한다. 개수보다 동작 품질과 상태 분포의 다양성이 중요하다. 사람 손으로 물체를 고쳐 놓는 장면은 녹화 중 포함하지 않는다.

저장 형식은 에피소드별 HDF5이며 다음을 포함한다.

- JPEG 카메라 프레임(기본 10 Hz)
- xArm5 관절 위치와 속도
- TCP 위치/쿼터니언
- 안전 필터 뒤에서 실제로 실행된 XYZ 속도 명령
- 각 센서 원본 타임스탬프와 기록 시각

녹화 중 노드가 종료되면 `.partial.h5`는 학습 대상에서 제외된다. 다음 녹화는 겹치지 않는 새 번호를 사용한다.

## 6. 데이터 검증과 눈으로 확인

학습 전에 모든 에피소드의 길이·NaN·시간 순서·속도 한계·JPEG 디코딩·센서 시간차를 검사한다.

```bash
ros2 run xarm_il validate_dataset -- \
  --data ~/xarm_il_data/xarm5_drag \
  --report ~/xarm_il_data/xarm5_drag/validation_report.json
```

0 명령만 있는 `dry_run` 데이터는 오류로 판정한다. 센서 시각 차이가 크다는 경고가 반복되면 카메라/ROS 시간 기준과 CPU 부하를 먼저 해결한다.

무작위 에피소드 몇 개를 MP4로 만들어 물체가 보이는지, 행동과 영상이 맞는지 확인한다.

```bash
ros2 run xarm_il visualize_episode -- \
  ~/xarm_il_data/xarm5_drag/episode_000000.h5 \
  --output ~/xarm_il_data/episode_000000_review.mp4
```

## 7. 정책 학습

모델 입력은 현재 모든 카메라 영상과 `[관절 위치 5, 관절 속도 5]`이고, 출력은 현재부터 미래 `horizon` 스텝까지의 XYZ 속도다. ResNet-18 영상 특징과 Transformer decoder를 사용한 행동 묶음 행동복제 모델이다. 원 논문의 전체 ACT/CVAE 구현과 동일하다고 주장하지 않는다.

학습/검증은 **프레임이 아니라 에피소드 단위**로 분리하여 인접 프레임 누출을 막는다. 정규화 통계도 학습 에피소드에서만 계산된다.

```bash
ros2 run xarm_il train_policy -- \
  --data ~/xarm_il_data/xarm5_drag \
  --output ~/xarm_il_runs/drag_v1 \
  --epochs 60 \
  --batch-size 16 \
  --horizon 20 \
  --pretrained-backbone
```

인터넷 없이 학습하거나 pretrained 가중치를 내려받을 수 없으면 `--pretrained-backbone`을 빼도 된다. GPU 메모리가 부족하면 `--batch-size 8`, CPU에서 데이터 로딩 문제가 있으면 `--workers 0`을 사용한다.

출력 파일은 다음과 같다.

- `best.pt`: 검증 손실이 가장 낮은 정책
- `last.pt`: 마지막 epoch 정책
- `history.jsonl`: epoch별 학습/검증 손실
- `run_config.json`: 데이터 분할, 모델 설정, 정규화 통계

저장·검증·데이터 로더·모델 역전파를 합성 데이터로 한 번에 시험할 수 있다.

```bash
ros2 run xarm_il smoke_test_pipeline
```

## 8. 정책 평가

학습에 쓰지 않은 물체 시작 위치를 미리 정하고 성공 조건을 명확히 한다. 예: “20초 안에 물체 전체가 바구니 안으로 들어감”. 먼저 로봇 출력이 차단된 상태로 실행한다.

```bash
ros2 launch xarm_il system_policy_eval.launch.py \
  robot_ip:=192.168.1.XXX \
  checkpoint:=/home/USER/xarm_il_runs/drag_v1/best.pt \
  dry_run:=true
ros2 service call /xarm_il/enable_policy std_srvs/srv/SetBool "{data: true}"
```

카메라·관절 입력 오류, 추론 지연, 출력 방향을 확인한다. 실제 시험은 빈 작업영역, 낮은 속도, 비상정지 준비, 1회씩 초기화한 조건에서만 `dry_run:=false`로 바꾼다.

```bash
ros2 launch xarm_il system_policy_eval.launch.py \
  robot_ip:=192.168.1.XXX \
  checkpoint:=/home/USER/xarm_il_runs/drag_v1/best.pt \
  dry_run:=false
ros2 service call /xarm_il/enable_policy std_srvs/srv/SetBool "{data: true}"
```

정책은 기본 15초 후 자동으로 idle이 된다. 카메라/관절 입력이 오래되거나 키보드 노드의 안전 heartbeat가 끊겨도 정지하며, 입력이 돌아와도 자동 재개하지 않으므로 서비스를 다시 호출해야 한다. 즉시 중단하려면 하드웨어 비상정지를 우선 사용하고, 정상 소프트웨어 정지는 Backspace/Esc 또는 다음 서비스를 사용한다.

```bash
ros2 service call /xarm_il/enable_policy std_srvs/srv/SetBool "{data: false}"
```

정책 실행 중 Shift를 누르면 키보드 모드로 넘어간다. F9만 눌러서는 비활성 정책이 켜지지 않으며, 서비스로 명시적으로 활성화해야 한다.

## 실험 결과 기록 권장안

졸업작품 평가는 훈련에 포함하지 않은 고정된 20~30개 초기 배치에서 수행한다.

- 성공률과 95% 신뢰구간
- 성공까지 걸린 시간
- 작업영역 제한 또는 사람 인계 횟수
- 손목 카메라만 사용한 모델과 고정 카메라를 추가한 모델 비교
- 시연 수(50/100/200개)에 따른 성공률

모델 손실만으로 로봇 성능을 판단하지 않는다. 검증 손실이 낮아도 카메라 밖 상태나 새로운 물체 위치에서 실패할 수 있다.

## 디렉터리

```text
src/xarm_il/
├── config/xarm5_il.yaml
├── launch/
│   ├── collect.launch.py
│   ├── policy_eval.launch.py
│   ├── system_collect.launch.py
│   ├── system_policy_eval.launch.py
│   └── teleop.launch.py
├── test/
└── xarm_il/
    ├── keyboard_teleop_node.py
    ├── safety_mux_node.py
    ├── episode_recorder_node.py
    ├── preflight_check.py
    ├── storage.py / dataset.py / validate_dataset.py
    ├── model.py / train_policy.py / policy_node.py
    └── visualize_episode.py / smoke_test_pipeline.py
```

## 현재 설계의 한계

- 회전 명령은 학습하지 않고 TCP 자세를 Servo가 유지하도록 하며 XYZ 이동만 다룬다.
- 직육면체 작업영역은 바구니, 책상, 로봇 링크의 정확한 충돌 형상을 대신하지 않는다.
- 최신 샘플 방식으로 10 Hz 기록하며 하드웨어 동기화 카메라는 아니다. 검증기가 시간차를 보고한다.
- 행동 복제는 시연에서 벗어난 상태에서 오차가 누적될 수 있다. 실패 상태를 사람이 키보드로 복구하는 고품질 시연을 추가 수집하는 방식(DAgger에 가까운 반복)을 후속 단계로 권장한다.
- 실제 xArm 및 카메라 드라이버가 연결된 Ubuntu 22.04 장비에서 최종 통합 시험이 반드시 필요하다.
