# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 명령어

```bash
# 의존성 설치
pip install -r requirements.txt

# Airflow DB 초기화 (로컬 테스트 실행 전 필수)
export AIRFLOW_HOME="${PWD}/.airflow"
airflow db init

# 전체 테스트 실행
pytest tests/test_dag_integrity.py -v --tb=short

# 단일 테스트 실행
pytest tests/test_dag_integrity.py::test_e2e_full_key_task_ids -v --tb=short
```

CI는 모든 push/PR 시 `.github/workflows/ci.yml`을 통해 Python 3.11 + SQLite 기반으로 실행됩니다.

## 아키텍처

Apache Airflow 기반의 **프로덕션급 E2E ML 서빙 플랫폼**입니다. 단순 모델 배포가 아닌 *안전한 모델 교체*를 목표로 하며, Promotion/Shadow 브랜칭, 배포 전 품질 게이트, 자동 롤백을 구현합니다.

**투-레포 설계:**
- **이 저장소 (`airflow-dags-dev`):** DAG 오케스트레이션 + 배포 정책
- **`mlops-infra-gitops`:** Kubernetes 인프라 (Helm, ArgoCD)
- **공유 상태:** S3 (피처/아티팩트) + MLflow Registry (모델 버전/alias)

### 핵심 DAG (`dags/`)

| DAG | 역할 |
|---|---|
| `e2e_full.py` | 메인 파이프라인: 데이터 → 학습 → 브랜치 → 배포 → 관측 → 롤백 |
| `dp_feature_pipeline.py` | 피처 추출, 검증, S3 저장 |
| `rollback_manual.py` | 긴급 수동 롤백 트리거 |
| `feast_materialize.py` | KubernetesPodOperator를 통한 Feature Store 구체화 |

### SSOT 설계 패턴

모든 식별자, 임계값, 변수 키는 두 개의 정규 파일에서만 정의합니다:

- **`dags/mlops_lib/core/ids.py`** — Task ID, XCom 키, 섀도우 사유 코드, TaskGroup 접미사 (`_S` 접미사 = TaskGroup 내부에서 사용하는 접미사 전용 변형)
- **`dags/mlops_lib/core/policy.py`** — 타임아웃, 품질 임계값, Airflow Variable 키, `Settings` 데이터클래스

`Settings.load()` 메서드는 런타임에 설정을 우선순위에 따라 결정합니다: 환경변수 > Airflow Variable > 하드코딩 기본값. `e2e_full.py` 상단의 `E2E` 별칭은 두 ID 클래스를 함께 임포트합니다.

**Task ID, XCom 키, 임계값을 절대 하드코딩하지 마세요.** 항상 `ids.py`와 `policy.py`를 참조하세요.

### Promotion / Shadow 브랜치 로직

```
학습 → branch_by_accuracy (BranchPythonOperator)
├─ accuracy ≥ promote_threshold AND drift 정상 → Promotion 경로
│   register → sensor → deploy → commit_current_json → reload → observe
├─ canary_threshold ≤ accuracy < promote_threshold AND drift 정상 → Canary 경로
│   Promotion과 동일 DAG 경로, XCOM_CANARY_TRAFFIC_PCT로 구분
└─ 그 외 → Shadow 경로 (deploy만, commit/reload 없음)
    사유: drift_detected | below_threshold | train_skipped | accuracy_invalid
```

드리프트 감지는 브랜치 전에 `mlops_lib/quality/drift_gate.py`로 실행됩니다 (KS-stat D > 0.20 = Shadow 강제). 브랜치 결과와 섀도우 사유는 XCom으로 하위 태스크에 전달됩니다.

### 태스크 조직 구조

- **`dags/e2e_full.py`** — 의존성 그래프만 정의; 비즈니스 로직 없음
- **`dags/pipelines/p_*.py`** — 태스크 콜러블 (train, drift, observe, register, reload, triton)
- **`dags/ml_code/`** — 핵심 ML 연산 (train_model, register_model, triton_deploy, triton_actions)
- **`dags/mlops_lib/`** — 공유 라이브러리: 품질 게이트, 관측성, 롤백, HTTP 유틸리티, SSOT

### 롤백 정책

롤백은 **최소 범위로 설계**됩니다: Triton 저장소 스냅샷 + `current.json`만 복원합니다. FastAPI 리로드 실패 시 저장소 SSOT는 롤백하지 않습니다 (공유 상태 복원은 파드 로컬 불일치보다 위험하기 때문). 롤백 태스크는 `[deploy, commit, observe]`에 `trigger_rule=ONE_FAILED`를 사용합니다.

### 관측성

배포 후 메트릭은 60–180초 윈도우 동안 Prometheus에서 수집됩니다. 임계값(에러율, p95 레이턴시)은 `policy.py`에 정의됩니다. 관측 실패 시 기본값은 "observe success"로 처리되어 오탐 롤백을 방지합니다.

### 외부 연동

- **Triton Inference Server** — ONNX 모델 서빙; 동적 배칭 설정은 `mlops_lib/core/triton_config.py`에서 생성
- **FastAPI** — 변형 라우팅 (Promotion/Shadow 분기)
- **MLflow Registry** — 모델 버전 관리 및 alias 관리
- **Prometheus** — 배포 후 메트릭 수집
- **S3/MinIO** — 피처 스토어 + 아티팩트 저장소
- **ArgoCD** — GitOps K8s 동기화
