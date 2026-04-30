# CONTINUE

https://is.muni.cz/auth/el/fi/jaro2026/PA228/index.qwarp?prejit=5526577

[ ] make it work even without mlflow credentials
[ ] set reasonable number of epochs
[ ] implement patience
[ ] fix versions in pyproject.toml
[x] use ModelCustom for inference
[x] rename ModelCutom to something meaningful
[x] remove the ModelLRASPP
[ ] prepare all required files
    [ ] all files in final_files
        [ ] model.pt → final_files/final_model.pt
        [ ] learning_curves.png → final_files/final_learning_curve.png
        [ ] model_architecture.png → final_files/final_model_architecture.png

    [ ] README
    [x] modify inference so it saves the outputs
[ ] cleanup the repo
    [ ] .venv
    [ ] __pycache__
    [ ] data_seg_public
    [ ] .env
    [ ] output_predictions
    [ ] models


## Optional
[ ] try improving miou:
    [ ] change learning rate
[ ] try improving learning of classess 3 and 6