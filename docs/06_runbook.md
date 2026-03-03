# Operational Runbook

## 1. Train 실패

- Airflow log 확인
- Slack 알림 확인
- Shadow 분기 여부 확인

---

## 2. Accuracy 미달

- threshold 확인
- Drift Gate 결과 확인
- Shadow 분기 정상 여부 확인

---

## 3. Deploy 실패

- triton_load 로그 확인
- snapshot 디렉토리 확인
- rollback_minimal 실행 여부 확인

---

## 4. AutoRollback 트리거

- Prometheus metric 확인
- 임계값 정책 확인
- Drift Gate 상태 확인

---

## 5. 수동 롤백

```
python3 rollback_manual.py --restore
```

---

## 6. Import 오류 발생 시

```
airflow dags list-import-errors
python3 -m compileall dags
```
