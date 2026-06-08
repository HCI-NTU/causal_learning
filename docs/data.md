# Data layout & split resolution

All datasets are COCO detection format except SODA (PASCAL VOC, converted).

```
data/MOCS/images/{train,val}
data/MOCS/instances_{train,val}.json
data/MOCS/instances_{train,val}_{earthmoving,foundation,superstructure}.json
data/CIS/images/{train,val,test}
data/CIS/instances_{train,val,test}.json
data/SODA/images/{train,test}
data/SODA/annotations/*.xml            # VOC -> instances_{train,test}.json
data/ACID/images/{train,test}
data/ACID/instances_{all,train,test}.json
data/ExtCon/images
data/ExtCon/extcon_gt.json
```

## Split-resolution rule
| dataset | has test? | resulting test target          | training use here |
|---------|-----------|--------------------------------|-------------------|
| MOCS    | no  | `val` (in-domain)                    | train (source)    |
| CIS     | yes | `test`  (`val`→`train` = trainval)   | target only       |
| ACID    | yes | `test`                               | target only       |
| SODA    | yes | `test`                               | target only (opt) |
| ExtCon  | n/a | whole set                            | target only       |

We train only on MOCS for both experiments, so the CIS trainval merge is encoded
for completeness / future single-source training.

## Label space
Model trains in the MOCS 13-class space (category_id 1..13). External GT and
predictions are remapped into that space at eval time; only shared classes are
scored. ExtCon covers all 13; CIS/ACID/SODA cover a subset — verify with
`python -m scripts.prepare_data --inspect`.

## Shared-class counts (from unified_class_schema.json)
The mapping is loaded from `ccd/data/unified_class_schema.json` (single source of
truth) and resolved by category *name*, so it is robust to id differences between
the schema and your json files.

- ExtCon : 13 — all MOCS classes (a few via synonyms: Tower crane=Static crane,
            Hanging hook=Hanging head, Vehicle crane=Crane, Pile driver=Pile driving)
- CIS    :  7 — Worker (people-helmet + people-no-helmet), Bulldozer (dozer),
            Excavator, Truck (dump-truck), Concrete mixer (mixer), Roller, Loader (wheel-loader)
            dropped: PC, PC-truck (no MOCS counterpart)
- ACID   :  8 — Static crane (tower_crane), Crane (mobile_crane), Roller (compactor),
            Bulldozer (dozer), Excavator, Truck (dump_truck), Loader (wheel_loader),
            Concrete mixer (cement_truck)
            dropped: backhoe_loader, grader (no MOCS counterpart)
- SODA   :  1 — Worker (person)
