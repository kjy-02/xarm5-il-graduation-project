# 노트북1 Codex 인수인계서

작성일: 2026-08-31  
프로젝트: xArm5 키보드 원격조작 기반 물체 끌기 모방학습

## 노트북1 Codex에게 처음 보낼 메시지

아래 문장을 그대로 전달하면 된다.

> 이 폴더는 xArm5 로봇으로 물체를 끌어 측면 바구니에 넣는 모방학습 졸업작품이다. 먼저 `HANDOFF_LAPTOP1.md`와 `README.md`를 끝까지 읽고, `src/xarm_il`의 현재 코드를 확인해라. 기존 안전장치와 데이터 형식을 유지하고 실제 장비에서 확인되지 않은 토픽·TF·작업영역을 임의로 추정하지 마라. 현재 오프라인 테스트는 통과했지만 Ubuntu 22.04 ROS 2 Humble 및 실제 xArm5 통합시험은 아직 하지 않았다. 우선 노트북2에서 카메라 토픽, 관절 이름, TF, MoveIt Servo 연결을 `dry_run=true`로 점검하는 과정을 도와주고, 발견한 실제 환경 정보에 맞춰 최소한의 설정만 수정해라. 수정할 때마다 정적 검사, 단위 테스트, 합성 파이프라인 테스트를 다시 실행하고 변경 내용과 장비 시험 결과를 이 인수인계서에 기록해라.

## 프로젝트 목표

- xArm5 그리퍼 끝부분을 이용하여 물체를 끈다.
- 물체를 로봇 측면에 고정된 바구니 안으로 떨어뜨린다.
- 사람 손으로 로봇을 직접 미는 방식 대신 키보드 원격조작으로 시연 데이터를 수집한다.
- 손목 아래 카메라 영상, 5축 관절 상태, 실제 실행된 TCP XYZ 속도를 저장한다.
- 서버에서 행동복제 모델을 학습한다.
- 학습된 체크포인트를 로봇 제어용 노트북2로 가져와 추론한다.

회전 행동과 그리퍼 개폐는 현재 학습 대상이 아니다. TCP 자세는 유지하고 XYZ 병진 이동만 사용한다.

## 컴퓨터별 역할

### 현재 데스크톱

- 최초 코드가 작성되고 오프라인 테스트된 장소다.
- ZIP을 만드는 원본 위치다.
- 이후 주 개발 원본은 노트북1의 Git 저장소로 전환하는 것을 권장한다.

### 노트북1: 개발 및 Codex 협업

- 코드 수정, 문서 관리, 실험 계획, 오류 분석을 담당한다.
- 노트북2에서 얻은 ROS 출력과 오류 로그를 받아 분석한다.
- 가능하면 Git 저장소의 기준 원본 역할을 맡는다.
- 실제 로봇 제어를 이 노트북에서 원격 네트워크로 실행하지 않는다.
- ROS 2가 설치되어 있지 않아도 일반 코드 편집은 가능하지만, ROS launch와 노드는 노트북2에서 최종 확인해야 한다.

### 노트북2: 로봇 및 카메라 제어

- Ubuntu 22.04, ROS 2 Humble, xarm_ros2, 카메라 드라이버를 실행한다.
- MoveIt Servo, 키보드 원격조작, 안전 mux, 데이터 기록을 실행한다.
- 학습된 정책의 실시간 추론도 노트북2에서 실행한다.
- 하드웨어 비상정지 장치를 작업자가 즉시 누를 수 있어야 한다.

정책 추론은 서버에서 원격으로 실행하지 않는다. 영상 왕복 지연이나 네트워크 단절 때문에 위험하므로, 서버에서는 오프라인 학습만 하고 `best.pt`를 노트북2로 복사한다.

### 서버: 오프라인 학습

- 노트북2에서 수집한 `episode_*.h5`를 전달받는다.
- 데이터 검증 후 PyTorch 모델을 학습한다.
- `best.pt`, `last.pt`, `history.jsonl`, `run_config.json`을 생성한다.
- 가장 좋은 `best.pt`를 노트북2로 돌려보낸다.
- ROS 2는 필수가 아니며 Python, PyTorch, torchvision, h5py, OpenCV가 필요하다.

## 현재 구현 상태

ROS 2 Python 패키지는 `src/xarm_il`에 있다.

구현 완료 항목:

- Shift 데드맨 기반 WASD/RF 키보드 조작
- teleop/policy/idle 명령 선택
- MoveIt Servo `/servo_server/delta_twist_cmds` 연결
- 속도 제한, TF 필수 확인, TCP 직육면체 작업영역 제한
- 키보드 heartbeat와 명령 timeout
- 종료 시 0 속도 명령 발행
- 사람의 Shift 입력에 의한 정책 제어권 인계
- 카메라·관절·TCP·실행 속도 HDF5 에피소드 기록
- `dry_run`, policy, idle, 센서 오류 상태에서 데이터 기록 차단
- 중복 센서 프레임 방지
- 데이터셋 무결성 검사 및 MP4 시각화
- ResNet-18 + Transformer 행동 묶음 모델
- 에피소드 단위 train/validation 분리
- 학습 통계, 체크포인트, 이력 저장
- 정책 입력 stale/NaN/추론 오류 시 비활성화
- 실제 xArm bring-up을 포함한 수집 및 정책 평가 launch
- 합성 데이터 기반 저장→검증→학습→체크포인트 재로딩 테스트

현재 검증 결과:

- Ruff 정적 검사 통과
- 전체 Python 문법 검사 통과
- 단위 테스트 17개 통과
- 합성 HDF5 및 1 epoch 학습 스모크 테스트 통과

아직 하지 않은 검증:

- 실제 Ubuntu 22.04 ROS 2 Humble에서 `colcon build`
- 실제 xArm5 연결
- 실제 카메라 토픽 연결
- `link_base -> link_eef` TF 확인
- 실제 키 방향과 로봇 이동 방향 확인
- 실제 작업영역 좌표 측정
- 실제 데이터 수집 및 모델 성능 평가

## 중요한 파일

- `README.md`: 설치부터 수집·학습·평가까지 전체 설명
- `src/xarm_il/config/xarm5_il.yaml`: 장비별 설정
- `src/xarm_il/launch/system_collect.launch.py`: xArm5와 수집 노드 통합 실행
- `src/xarm_il/launch/system_policy_eval.launch.py`: xArm5와 정책 평가 통합 실행
- `src/xarm_il/xarm_il/keyboard_teleop_node.py`: 키보드 입력
- `src/xarm_il/xarm_il/safety_mux_node.py`: 최종 안전 명령 필터
- `src/xarm_il/xarm_il/episode_recorder_node.py`: HDF5 데이터 기록
- `src/xarm_il/xarm_il/validate_dataset.py`: 데이터 검사
- `src/xarm_il/xarm_il/train_policy.py`: 서버 학습
- `src/xarm_il/xarm_il/policy_node.py`: 노트북2 실시간 추론
- `src/xarm_il/test`: 단위 테스트

## 실제 장비에서 반드시 확인할 설정

`src/xarm_il/config/xarm5_il.yaml`의 다음 값은 예시일 뿐이다.

1. `camera_topics`
   - 실제 손목 카메라의 `sensor_msgs/msg/Image` 토픽으로 변경한다.
   - `CompressedImage` 토픽을 그대로 넣으면 동작하지 않는다.

2. `joint_names`
   - `/joint_states` 출력과 정확히 같아야 한다.
   - 기본 예상값은 `joint1`부터 `joint5`까지다.

3. `command_frame`, `eef_frame`
   - 기본값은 `link_base`, `link_eef`다.
   - xArm prefix 또는 커스텀 URDF를 사용하면 달라질 수 있다.

4. `workspace_min`, `workspace_max`
   - 현재 값은 임시값이다.
   - 실제 책상, 바구니, 카메라, 로봇 자세를 기준으로 직접 측정해야 한다.

5. `axis_signs`
   - W/D/R이 의도한 +X/+Y/+Z 방향인지 `dry_run`과 매우 낮은 속도로 확인한다.

6. `robot_ip`
   - launch 인자로 실제 xArm 컨트롤러 IP를 전달한다.

## 노트북2 첫 실행 순서

작업공간 예시:

```bash
mkdir -p ~/xarm5_il_ws/src
cp -r /압축해제경로/src/xarm_il ~/xarm5_il_ws/src/xarm_il
cd ~/xarm5_il_ws
```

의존성과 xarm_ros2 설치 방법은 `README.md`를 따른다. 빌드 후:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

실제 토픽과 TF를 먼저 확인한다.

```bash
ros2 topic list | grep -E 'image_raw|joint_states'
ros2 topic echo /joint_states --once
ros2 run tf2_ros tf2_echo link_base link_eef
```

카메라를 실행하고 첫 launch는 반드시 다음처럼 한다.

```bash
ros2 launch xarm_il system_collect.launch.py \
  robot_ip:=192.168.1.XXX dry_run:=true
```

다른 터미널에서:

```bash
source ~/xarm5_il_ws/install/setup.bash
ros2 run xarm_il preflight_check -- \
  --camera-topic /실제/카메라/image_raw --duration 5
```

프리플라이트 결과와 다음 명령의 출력을 노트북1 Codex에게 전달한다.

```bash
ros2 topic info /servo_server/delta_twist_cmds --verbose
ros2 service list | grep servo
ros2 node list
ros2 topic echo /xarm_il/safety_status --once
```

`dry_run=true`에서는 로봇이 절대 움직이지 않아야 하고 `/xarm_il/executed_twist`가 0이어야 한다. 이 확인이 끝난 뒤 작업영역을 비우고 속도를 낮춰 `dry_run=false`로 전환한다.

## 데이터와 모델 이동 흐름

```text
노트북2
  성공 시연 수집
  ~/xarm_il_data/xarm5_drag/episode_*.h5
        │
        ├── 검증 및 영상 확인
        │
        └──────────────> 서버
                          오프라인 학습
                          best.pt 생성
                               │
                               └────────> 노트북2
                                           dry_run 정책 평가
                                           실제 저속 평가

노트북1
  코드 기준 원본, Codex 분석, 설정/문서/실험 기록 관리
```

노트북2에서 데이터를 서버로 보내기 전에:

```bash
ros2 run xarm_il validate_dataset -- \
  --data ~/xarm_il_data/xarm5_drag \
  --report ~/xarm_il_data/xarm5_drag/validation_report.json
```

서버에서는 ROS 없이 다음과 같이 실행할 수 있다.

```bash
cd /프로젝트/압축해제경로
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r src/xarm_il/requirements-ml.txt
python3 -m pip install "opencv-python-headless>=4.6,<4.12"
python3 -m pip install -e src/xarm_il

validate_dataset --data /데이터/xarm5_drag
train_policy \
  --data /데이터/xarm5_drag \
  --output /학습결과/drag_v1 \
  --epochs 60 --batch-size 16 --horizon 20 --pretrained-backbone
```

GPU 서버에서는 PyTorch를 서버 CUDA 버전에 맞춰 먼저 설치한다. `requirements-ml.txt`는 Ubuntu ROS/OpenCV 호환성을 위해 NumPy를 1.x로 제한한다. ROS가 없는 서버에서는 위 명령처럼 NumPy 1.x를 지원하는 `opencv-python-headless<4.12`를 별도로 설치한다.

학습 후 노트북2로 최소한 다음을 복사한다.

- `best.pt`
- `run_config.json`
- `history.jsonl`
- 사용한 데이터셋의 `validation_report.json`

## 정책 평가 위치

정책 파일을 노트북2로 복사하고 먼저:

```bash
ros2 launch xarm_il system_policy_eval.launch.py \
  robot_ip:=192.168.1.XXX \
  checkpoint:=/절대경로/best.pt \
  dry_run:=true
```

입력이 모두 준비된 뒤 별도 터미널에서:

```bash
ros2 service call /xarm_il/enable_policy \
  std_srvs/srv/SetBool "{data: true}"
```

카메라나 관절 입력이 끊기면 정책은 비활성화되고 자동 재개하지 않는다. 다시 시험하려면 원인을 해결하고 서비스를 다시 호출한다. 실제 평가 전에는 반드시 `dry_run=true`에서 출력 방향과 추론 지연을 확인한다.

## 절대 유지해야 할 안전 원칙

- `dry_run` 기본값을 false로 바꾸지 않는다.
- 안전 mux를 우회하여 `/servo_server/delta_twist_cmds`에 직접 발행하지 않는다.
- 공식 `xarm_keyboard_input`을 동시에 실행하지 않는다.
- 공식 xArm Servo launch에는 joystick 노드도 있으므로 게임패드와 SpaceMouse를 분리하고 별도 `/joy` 발행 노드를 실행하지 않는다.
- 로봇 주변에 사람이 있을 때 정책을 켜지 않는다.
- 하드웨어 비상정지는 소프트웨어 정지보다 우선한다.
- TF나 카메라가 없을 때 임시로 안전 검사를 끄지 않는다.
- 실제 좌표를 측정하기 전에 작업영역을 넓히지 않는다.
- 정책 추론을 서버나 불안정한 무선 네트워크 너머에서 실행하지 않는다.
- 수집과 실행 사이에 카메라 위치, 렌즈, 해상도, 조명을 함부로 바꾸지 않는다.

## 손목 카메라 관련 주의

코드는 손목 카메라 한 대로 동작하지만, 물체나 바구니가 영상에서 사라지면 학습 문제가 부분관측 상태가 된다. 다음을 만족하지 못하면 고정된 전체 시점 USB 카메라를 추가하는 것을 권장한다.

- 끌기 시작부터 투입까지 물체가 거의 계속 보인다.
- 바구니 방향 또는 고정 표식이 보인다.
- 로봇 관절 상태만으로 카메라의 전역 위치를 해석할 수 있도록 카메라 마운트가 고정되어 있다.

카메라를 추가하면 수집과 추론 설정의 `camera_names`, `camera_topics` 순서가 완전히 같아야 한다. 서로 다른 카메라 구성의 에피소드는 현재 검증기와 데이터 로더가 거부한다.

## ZIP 이동 후 권장 관리 방법

ZIP은 최초 전달에만 사용하고 이후에는 Git을 사용하는 것이 좋다.

압축을 풀 폴더 이름은 한글 대신 `xarm5_il_project`처럼 영문으로 두면 Ubuntu 셸 명령과 서버 전송이 편하다. ZIP을 만든 뒤 데스크톱과 두 노트북에서 SHA-256을 비교하여 전송 중 손상이나 서로 다른 버전 사용을 방지한다.

Windows PowerShell:

```powershell
Get-FileHash .\xarm5_il_project_2026-08-31.zip -Algorithm SHA256
```

Ubuntu:

```bash
sha256sum xarm5_il_project_2026-08-31.zip
```

노트북1에서 압축을 푼 직후:

```bash
git init
git add README.md HANDOFF_LAPTOP1.md .gitignore src/xarm_il
git commit -m "Initial xArm5 imitation learning pipeline"
```

가능하면 학교 Git 서버나 비공개 GitHub 저장소를 원격 저장소로 사용한다. 실제 데이터셋과 체크포인트는 용량이 크므로 Git에 넣지 않는다.

Git에 넣지 않을 항목:

- `episode_*.h5`
- `*.partial.h5`
- `best.pt`, `last.pt`
- `build/`, `install/`, `log/`
- `.venv/`, `__pycache__/`

데이터셋과 모델 파일은 별도 저장장치 또는 서버 디렉터리에 두고 날짜, 설정 버전, SHA-256 체크섬을 함께 기록한다.

## 네트워크 관련 권장사항

- 노트북2의 xArm 유선 Ethernet과 학교/서버 네트워크를 가능하면 별도 인터페이스로 사용한다.
- xArm 전용 NIC에 잘못된 기본 게이트웨이를 설정해 인터넷/서버 트래픽이 로봇망으로 향하지 않게 한다.
- 학교 네트워크에서 다른 ROS 2 사용자의 DDS와 섞이지 않도록 고유한 `ROS_DOMAIN_ID`를 정한다.
- ROS 그래프는 가능하면 노트북2 내부에만 둔다. 노트북1과 서버가 ROS 토픽을 직접 구독할 필요는 없다.
- 카메라와 ROS 타임스탬프 차이가 크면 NTP/chrony와 카메라 드라이버 시간 설정을 확인한다.

예시:

```bash
export ROS_DOMAIN_ID=37
```

이 값은 노트북2에서 실행하는 모든 ROS 터미널에 동일하게 적용한다.

## 첫 실험 목표

처음부터 수백 개를 수집하지 않는다.

1. 실제 장비 `dry_run` 프리플라이트 통과
2. 아주 낮은 속도로 키 방향과 작업영역 차단 확인
3. 성공 시연 10개 수집
4. 검증기 통과 및 MP4 육안 확인
5. 서버에서 1~3 epoch 시험 학습
6. `best.pt`를 노트북2로 복사
7. 정책 `dry_run` 추론 확인
8. 문제가 없으면 성공 시연 30~50개로 전체 흐름 재확인
9. 이후 100~300개 규모의 본 데이터 수집

## 노트북1 Codex가 다음에 해야 할 일

1. 임의로 코드를 더 확장하기 전에 실제 노트북2의 토픽/TF/빌드 결과를 받는다.
2. `colcon build` 오류가 있으면 전체 오류 로그와 설치된 xarm_ros2 branch를 확인한다.
3. 실제 카메라 토픽, 해상도, FPS, 관절 이름을 이 문서에 기록한다.
4. 실제 TCP 안전 범위를 측정한 뒤 YAML만 최소 수정한다.
5. 변경 후 단위 테스트와 `smoke_test_pipeline`을 재실행한다.
6. 실제 수집 데이터 1개를 `validate_dataset`과 `visualize_episode`로 확인한다.
7. 10개 파일로 서버 시험 학습을 하고 체크포인트가 노트북2에서 로딩되는지 확인한다.
8. 성공률 실험 전 코드와 설정을 Git commit/tag로 고정한다.

실제 장비에서 얻은 정보는 이 문서 아래에 날짜와 함께 계속 추가한다.

## 실제 장비 확인 기록

아직 기록 없음.

추후 기록 형식:

```text
날짜:
xarm_ros2 branch/commit:
노트북2 Ubuntu/ROS 버전:
로봇 IP:
카메라 모델:
카메라 토픽/해상도/FPS:
joint_names:
command_frame/eef_frame:
workspace_min/max:
dry_run 프리플라이트 결과:
실제 저속 이동 결과:
발견한 오류와 수정 commit:
```
