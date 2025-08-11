# 🐛 Debug Tools

## 📊 분석 도구들

### `training_data_analysis.py`
**용도**: 학습 데이터의 twist vector 통계 분석 (평균, 표준편차, 범위)
**사용법**: 
```bash
cd packages/policy/v1
python debug/training_data_analysis.py
```
**출력**: `debug/training_data_analysis.png` - 히스토그램 차트

### `velocity_field_analysis.py`
**용도**: 모델이 예측하는 twist vector의 크기와 방향 분석
**사용법**:
```bash
cd packages/policy/v1
python debug/velocity_field_analysis.py
```
**출력**: `debug/velocity_field_analysis_result.json` - 수치 분석 결과

### `velocity_field_visualization.py`
**용도**: 모델로 생성된 벡터장 시각화 (2D 그리드 상 화살표)
**사용법**:
```bash
cd packages/policy/v1
python debug/velocity_field_visualization.py
```
**출력**: `debug/velocity_field_visualization.png` - 벡터장 시각화

## 📝 Notes
- 모든 스크립트는 `packages/policy/v1` 디렉토리에서 실행
- 정규화된 모델 기준으로 작성됨
- 분석 결과는 `debug/` 폴더에 저장

