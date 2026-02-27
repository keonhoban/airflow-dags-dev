# GitOps 기반 E2E ML Platform (Production-Oriented Design)

Production-grade ML Platform을 **GitOps 중심으로 설계·구축·운영**한 프로젝트입니다.

단순 모델 배포가 아니라, 운영 환경에서의 **모델 교체 리스크 통제, 분기 전략, 롤백 정책, 재현성 보장**까지 포함한 플랫폼 구조를 구현했습니다.

핵심 특징:

- GitOps(ArgoCD) 기반 **dev/prod 완전 분리**
- Airflow 기반 **E2E 자동화 파이프라인**
- Accuracy 기준 **Promotion / Shadow 전략**
- MLflow Registry + Alias 기반 모델 관리
- Triton Inference Server 연동
- FastAPI Reload 연계
- Minimal Rollback 정책
- SSOT(IDs/Policy) 설계로 유지보수 안정성 확보
- Airflow DAG Import Error 0

---

# 1. Architecture Overview

## End-to-End Flow

```
Data → Feature → Train → Registry → Branch
        ↓
   (Promotion) → Deploy → Smoke Test → Commit → Reload
        ↓
   (Shadow) → Deploy(shadow) → Reload
        ↓
      Failure → Minimal Rollback
```

## 주요 구성 요소

| Layer | Component | 역할 |
| --- | --- | --- |
| Orchestration | Airflow | E2E 파이프라인 제어 |
| Model Registry | MLflow | 모델 버전 및 Alias 관리 |
| Serving | Triton | 모델 로딩/추론 |
| API | FastAPI | 모델 엔드포인트 및 reload |
| Infra | Kubernetes + GitOps | 배포 및 환경 분리 |

---

# 2. 설계 철학 (Design Principles)

## 2.1 SSOT (Single Source of Truth)

- `mlops_lib/core/ids.py`
- `mlops_lib/core/policy.py`

모든 task_id, XCom key, 정책 상수는 중앙 정의.

### 목적

- 문자열 하드코딩 제거
- TaskGroup 포함 task_id mismatch 방지
- 리팩토링 시 영향 범위 최소화
- 운영 중 오타/rename 사고 방지

포인트:

> DAG 엔트리포인트는 orchestration만 담당하고, 식별자와 정책은 분리하여 변경 안전성을 확보했습니다.
> 

---

## 2.2 Promotion / Shadow 전략

Accuracy 기준 분기:

- `accuracy >= threshold` → Promotion
- `accuracy < threshold` → Shadow
- train 실패 → Shadow

### Promotion Path

- MLflow register + alias
- Sensor로 ready 확인
- Triton materialize (version 기반)
- Smoke test
- Commit current.json
- FastAPI reload

### Shadow Path

- run_id 기반 materialize
- 운영 모델 영향 없음
- FastAPI shadow reload

---

## 2.3 Rollback 정책

### Minimal Rollback 원칙

- Triton repository state만 복구
- current.json snapshot 기반 복원
- 실패 디렉토리 격리
- FastAPI reload 실패는 repo rollback 안 함

왜?

> 모델 repository SSOT를 되돌리는 것이 reload 실패보다 더 위험하기 때문입니다.
> 

---

## 2.4 DAG 책임 분리

| Layer | 역할 |
| --- | --- |
| e2e_full.py | DAG 정의 및 의존성 연결 |
| pipelines/full_e2e.py | orchestration callables |
| ml_code/* | 실제 비즈니스 로직 |
| core/* | SSOT (ids / policy) |

→ DAG 정의와 구현을 완전히 분리

---

# 3. 운영 안정성 전략

## 3.1 Import 안정성

- `airflow dags list-import-errors = 0`
- `python3 -m compileall dags` 통과

## 3.2 Sensor 전략

- mode="reschedule"
- timeout 정책 분리
- 모델 Ready 상태 검증

## 3.3 Smoke Test

- Triton load → ready → inference smoke
- 배포 실패 시 rollback_minimal 트리거

---

# 4. 실패 시나리오

### Case 1: Train 실패

- accuracy None
- Shadow 분기
- 운영 모델 영향 없음

### Case 2: Accuracy 미달

- Shadow
- Slack 알림
- 운영 모델 유지

### Case 3: Deploy 실패

- rollback_minimal 실행
- snapshot 기반 복구

---

# 5. 운영 확장 고려 사항

- Accuracy threshold는 Airflow Variable로 조정 가능
- model_name / alias 변경 가능
- GPU Triton 전환 시 구조 유지
- 다중 모델 확장 시 TG 확장 가능

---

# 6. Why This Is Production-Oriented

이 프로젝트는 단순 데모가 아니라 다음을 고려하여 설계되었습니다:

- 모델 교체 시 리스크 통제
- 실패 격리
- 정책 중앙화
- 재현 가능성
- 변경 안전성
- GitOps 기반 환경 일관성

---

# 7. Proof of Stability

- Airflow Import Error 0
- Shadow/Promotion 분기 정상 동작
- Rollback 정상 동작
- SSOT 기반 task_id 안정성

---

# 8. 기술 스택

- Kubernetes
- ArgoCD (GitOps)
- Airflow
- MLflow
- Triton Inference Server
- FastAPI
- Python

---

# 9. 프로젝트의 목적

> 
> 
> 
> 모델을 단순히 배포하는 것이 아닌,
> 
> 모델 교체를 운영 환경에서 안전하게 수행할 수 있는 플랫폼을 설계하는 것
>
