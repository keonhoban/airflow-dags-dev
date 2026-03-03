# Deployment Flow

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
