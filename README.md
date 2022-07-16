# spacenet8_solution

## data preprocess

```
./scripts/preprocess.sh
```

## train foundation models

```
echo "WANDB_API_KEY = {YOUR_WANDB_API_KEY}" > .env
python tools/train_net.py --type building
```
