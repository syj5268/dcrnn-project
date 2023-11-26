# dcrnn-project

## Preprocessing
`generate_training_data.py` : speed data 전처리 <br>
`gen_adj_mx.py` : adjacency matrix 전처리

## Model
`model/dcrnn_cell.py` <br>
`model/dcrnn_model.py` <br>
`model/supervisor.py` <br> 
`eval_baseline_methods.py` : Static, HA, VAR 비교

## Input
`data/model/dcrnn_core_test.yaml` : urban-core 데이터 학습 및 테스트를 위한 파라미터

## Output
`run_pytorch.py` : 모델 학습 <br>
`appendix.ipynb` : 결과값 확인

Reference: https://github.com/chnsh/DCRNN_PyTorch
