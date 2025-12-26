## 개요 (KOR)
본 릴리즈는 **보건의료 빅데이터 개방포털 업무안내·FAQ 기반 한국어 Q&A 응용 모델(LoRA 어댑터)** 배포용 번들입니다.  
⚠️ **베이스 모델 가중치는 포함하지 않습니다.** (사용자 환경에서 별도 준비 필요)

## 포함 파일 (KOR)
- LoRA 어댑터: `outputs/hira_lora_20251217_001/final_model/`
  - `adapter_model.bin`, `adapter_config.json`
  - `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `README.md`
- 학습 코드: `train/`
- 실행/서비스 스크립트: `solar_web_faq70.py`
- 정책 문서: `LICENSE`, `USAGE_POLICY.md`, `NOTICE.md`, `.gitignore`

## 사용 방법 (KOR)
1) 릴리즈 자산(zip) 다운로드 후 압축 해제  
2) 베이스 모델을 별도로 준비(미포함)한 뒤, 어댑터를 결합하여 사용합니다.  
- `model/` 폴더는 비어있습니다. 사용자가 베이스 모델을 이 경로에 배치하는 구성을 권장합니다.  
- 어댑터 경로: `./outputs/hira_lora_20251217_001/final_model`

## 라이선스 (KOR)
- 저작권: 건강보험심사평가원(HIRA)
- 라이선스: CC BY-NC 4.0
- **상업적 이용 금지 / Commercial license is not provided.**
- 상세: `LICENSE`, `USAGE_POLICY.md`, `NOTICE.md`

---

## Overview (ENG)
This release provides a **Korean Q&A application model (LoRA adapter)** bundle.  
⚠️ **Base model weights are NOT included** and must be obtained separately by the user.

## How to use (ENG)
- Adapter path: `./outputs/hira_lora_20251217_001/final_model`

## License (ENG)
- Copyright: HIRA
- License: CC BY-NC 4.0
- **Commercial use is prohibited / No commercial license is provided.**
