# GitOps 기반 Production-Grade E2E ML Platform

이 프로젝트는 모델을 단순히 배포하는 것이 아니라,

> 모델 교체를 운영 환경에서 안전하게 수행하기 위한 ML Platform

을 목표로 설계되었습니다.

---

# 1. What Problem This Solves

운영 환경에서 모델 교체는 다음과 같은 리스크를 가집니다:

- 정확도 저하
- 데이터 분포 변화
- 잘못된 모델 로딩
- 배포 실패 후 불완전 상태
- 과잉 롤백

이 플랫폼은 이러한 리스크를 통제하기 위해 다음을 구현합니다:

- Accuracy + Drift Gate 기반 Promotion / Shadow 분기
- Drift Gate 사전 차단
- Sensor 기반 Ready 검증
- Smoke Test 기반 배포 검증
- Minimal Rollback 정책
- SSOT 설계로 변경 안전성 확보

---

# 2. Core Design Decisions

## 2.1 Orchestration-Only DAG

`dags/e2e_full.py`는 오직 의존성 그래프만 정의합니다.

비즈니스 로직은:

- `pipelines/*`
- `ml_code/*`

로 완전 분리.

→ DAG 수정 시 영향 범위 최소화  
→ 유지보수 안전성 확보  

---

## 2.2 SSOT (Single Source of Truth)

모든 식별자/정책은 중앙 정의:

- `mlops_lib/core/ids.py`
- `mlops_lib/core/policy.py`

목적:

- 문자열 하드코딩 제거
- TaskGroup prefix mismatch 방지
- rename/refactor 안전성 확보
- 운영 사고 방지

---

## 2.3 Promotion / Shadow 전략

Accuracy + Drift Gate 기반 분기:

- accuracy ≥ threshold → Promotion
- accuracy < threshold → Shadow
- train 실패 → Shadow
- drift 감지 → 강제 Shadow

운영 모델은 Promotion이 아닌 이상 변경되지 않습니다.

---

## 2.4 Minimal Rollback 원칙

Rollback은 다음만 수행합니다:

- Triton repository snapshot 복구
- current.json 복원
- 실패 디렉토리 격리

하지 않는 것:

- FastAPI reload 실패 시 repo rollback

이유:

> Repository SSOT를 되돌리는 것이 reload 실패보다 더 위험하기 때문입니다.

---

# 3. End-to-End Flow
```
Data → Feature → Train → Branch
↓
(Promotion) → Register → Ready Sensor → Deploy → Smoke → Commit → Reload → Observe
↓
(Shadow) → Deploy(shadow) → Observe
↓
Failure → Minimal Rollback
```


---

# 4. Observability & Auto Rollback

배포 후:

- Prometheus 기반 metric 수집
- AutoRollback 정책 평가
- 임계값 초과 시 Airflow task 실패
- DAG trigger_rule=ONE_FAILED로 rollback 실행

> 관측 실패는 무조건 롤백이 아니라, 정책 기반 판단 결과에 따라 롤백

---

# 5. Stability Proof

- Airflow Import Error 0
- Promotion / Shadow 분기 정상 동작
- Sensor 기반 Ready 확인
- Smoke Test 실패 시 rollback 정상 실행
- SSOT 기반 task_id 안정성 확보

---

# 6. Stack

- Kubernetes
- ArgoCD (GitOps)
- Airflow
- MLflow
- Triton Inference Server
- FastAPI
- Prometheus

