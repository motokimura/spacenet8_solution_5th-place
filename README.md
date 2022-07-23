# spacenet8_solution

## data preprocess

```
./scripts/preprocess.sh
```

## train networks

```
echo "WANDB_API_KEY = {YOUR_WANDB_API_KEY}" > .env
python tools/train_net.py --task {task} --exp_id {exp_id}
```

## test networks

```
python tools/test_net.py --exp_id {exp_id}
```