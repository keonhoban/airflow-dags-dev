# Operational Runbook

## 1. Train 실패

- Airflow log 확인
- Slack 알림 확인
- Shadow 분기 여부 확인

```bash
# Airflow UI에서 태스크 로그 확인
airflow tasks logs e2e_full train_and_evaluate <execution_date>

# Pod 상태 확인
kubectl get pods -n airflow-dev -l dag_id=e2e_full
kubectl logs -n airflow-dev <pod-name> --tail=100
```

Train 실패 시 `branch_by_accuracy`는 자동으로 Shadow 경로를 선택합니다 (사유: `train_skipped`).

---

## 2. Accuracy 미달

- threshold 확인: `policy.py` → `VAR_ACCURACY_THRESHOLD` (기본값: `0.60`)
- Drift Gate 결과 확인
- Shadow 분기 정상 여부 확인

```bash
# 현재 accuracy threshold 확인
airflow variables get accuracy_threshold

# Drift Gate 임계값 확인
airflow variables get drift_ks_stat_threshold  # 기본값: 0.20
```

Accuracy < threshold이면 Shadow 분기됩니다. Drift Gate에서 KS-stat D > 0.20이면 강제 Shadow입니다.

---

## 3. Deploy 실패

- triton_load 로그 확인
- snapshot 디렉토리 확인
- rollback_minimal 실행 여부 확인

```bash
# Triton pod 로그 확인
kubectl logs -n triton-dev deployment/triton-dev --tail=100

# NFS 모델 디렉토리 확인
kubectl exec -n triton-dev deployment/triton-dev -- ls -la /models/

# Triton 모델 상태 확인
curl -s http://<triton-host>:8000/v2/models/best_model | jq .
```

Deploy 태스크 실패 시 `trigger_rule=ONE_FAILED`로 `rollback_minimal`이 자동 트리거됩니다.

---

## 4. AutoRollback 트리거

- Prometheus metric 확인
- 임계값 정책 확인: `policy.py` 참조
- Drift Gate 상태 확인

```bash
# Prometheus에서 에러율 직접 쿼리
curl -s 'http://<prometheus>:9090/api/v1/query?query=rate(fastapi_requests_total{status=~"5.."}[1m])' | jq .

# 현재 임계값 확인
airflow variables get observe_error_rate_threshold    # 기본값: 0.02 (2%)
airflow variables get observe_latency_p95_threshold_sec  # 기본값: 0.8초
```

관측 윈도우: `observe_window_sec` (기본값: 180초). 에러율 또는 p95 레이턴시가 임계값을 초과하면 롤백 트리거됩니다.

---

## 5. 수동 롤백

```bash
cd dags && python3 rollback_manual.py --restore
```

또는 Airflow UI에서 직접 트리거:

```bash
airflow dags trigger rollback_manual
```

수동 롤백 수행 내용:
- `current.json` 스냅샷 복원
- 실패 버전 디렉토리 격리 (quarantine)
- Triton unload → load 이전 버전

---

## 6. Import 오류 발생 시

```bash
airflow dags list-import-errors
python3 -m compileall dags
```

Import 오류가 있으면 DAG가 Airflow 스케줄러에 등록되지 않습니다. CI에서도 `test_dag_integrity.py`로 검증합니다.
