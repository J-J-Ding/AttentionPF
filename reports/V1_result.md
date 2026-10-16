# V1 ETTh1 Interval Forecast Result

This tracked file is the initial V1 report placeholder. A completed Stage-II
run overwrites it automatically with the experiment ID, Stage-I checkpoint
provenance, adaptation settings, parameter statistics, resource/timing data,
overall metrics, and metrics by forecast horizon.

Run Stage I first:

```powershell
python pretrain.py --config configs/v1.yaml
```

Then set `pretrain.checkpoint` to its `best.pt`, choose `adaptation.mode` from
`full`, `qk`, or `qkv`, and run:

```powershell
python run.py --model v1 --config configs/v1.yaml
```

No numerical result is pre-filled here because unexecuted metrics would not be
reproducible evidence.
