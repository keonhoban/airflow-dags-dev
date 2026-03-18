# Deployment Flow

## Drift Gate — KS-statistic 설계 근거

Promotion 경로 진입 전에 `drift_gate` 태스크가 실행된다.
새 feature 분포와 마지막 학습에 사용된 reference 분포를 비교해
드리프트가 감지되면 promotion을 차단하고 shadow 경로로 우회한다.

### 왜 KS-statistic(D)인가?

Kolmogorov-Smirnov 통계량 D는 두 분포의 누적 분포 함수(CDF) 사이 최대 절대 차이를 나타낸다.
D ∈ [0, 1] 이며, 0이면 두 분포가 동일, 1이면 완전히 분리된 분포다.

**p-value를 사용하지 않는 이유:**
KS-test의 p-value는 샘플 수 n에 강하게 의존한다.
동일한 D=0.05라도 n=100이면 p=0.7(유의하지 않음), n=10,000이면 p≈0(매우 유의)이 된다.
배치 데이터 파이프라인에서는 실행마다 row 수가 다를 수 있으므로,
p-value threshold를 고정하면 데이터 볼륨이 늘어날수록 false-positive(불필요한 shadow 우회)가 급증한다.
반면 D값은 샘플 수에 관계없이 분포 차이 자체를 나타내므로 threshold 튜닝이 안정적이다.

### 왜 threshold = 0.20인가?

D=0.20은 두 분포의 최대 CDF 차이가 20%임을 의미한다.
이 수준이면 모델이 학습 시점에 보지 못한 입력 영역이 유의미하게 존재한다고 판단할 수 있다.

- D < 0.10: 정상 범주. 자연스러운 일별 분포 변동 수준.
- D 0.10~0.20: 경계 구간. 모니터링 강화 대상이지만 배포는 허용.
- D > 0.20: 프로모션 차단. shadow 배포로 우회해 운영 모델 영향 없이 검증.

threshold는 Airflow Variable `drift_ks_stat_threshold`로 런타임에 조정 가능하다.
초기값 0.20은 사내 feature 분포의 일별 변동 D를 측정한 결과를 기반으로 설정했으며,
feature가 추가되거나 데이터 소스가 변경될 때 재조정이 필요하다.

### 구현 참고

- 비교 대상: 신규 feature batch vs. S3 `latest/` 경로의 reference features
- 샘플 수: 최대 `drift_sample_n`개 (기본 2000), 수치 컬럼만 대상
- 외부 의존성: numpy만 사용 (scipy 불필요, 재현성 보장)
- 코드 위치: `dags/mlops_lib/quality/drift_gate.py`

---

## Promotion Path

1. Train
2. Accuracy ≥ threshold
3. MLflow register + alias
4. Sensor ready 확인
5. Triton materialize(version)
6. Smoke test
7. Commit current.json
8. FastAPI reload
9. Observe metrics

---

## Shadow Path

1. Train 실패 or accuracy 미달
2. run_id 기반 materialize
3. FastAPI shadow reload
4. 운영 모델 영향 없음

---

## Failure Path

Deploy or Observe 실패 → rollback_minimal 실행
