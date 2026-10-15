# Datasets

Raw files live in `datasets/raw/`. V0 expects an ETT-style CSV containing a
parseable `date` column followed by finite numeric feature columns.

The supplied baseline dataset is `datasets/raw/ETTh1.csv`, with columns:

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

The time column is used for ordering and validation only. In `MS` mode, the
prediction target must already be the final numeric feature (`OT` here). The
loader does not infer or reorder target columns.

