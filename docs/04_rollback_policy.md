# Rollback Policy

## 1. 목적

배포 실패 시 플랫폼을 안전한 상태로 복구한다.

---

## 2. 언제 롤백하는가

- Triton load 실패
- Triton ready 실패
- Smoke test 실패
- Post-deploy observability 정책 위반

---

## 3. 무엇을 롤백하는가

- Triton model repository snapshot 복원
- current.json 복구

---

## 4. 무엇은 롤백하지 않는가

- FastAPI reload 실패는 repository rollback 하지 않음

이유:

- Repository는 SSOT
- Reload 실패는 애플리케이션 레벨 문제
- Repository 되돌림은 더 큰 리스크

---

## 5. 실패 격리 전략

- 실패 디렉토리는 quarantine 처리
- snapshot 기반 복구
- 수동 rollback_manual.py 지원

---

## 6. 설계 철학

Rollback은 "마지막 수단"이며,
과잉 롤백을 방지하는 것이 안정성의 핵심
