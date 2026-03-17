# Coverage / Limitations

Coverage levels reflect how confidently the docs can describe output shape from source:

- `declared`: derived from explicit schema maps or provider column constants
- `partially-derived`: source gives useful clues but not a complete column contract
- `live-observed`: observed in a captured sample snapshot
- `not-available`: no source-derived or live-observed output contract was found

## Counts

- `declared`: 248
- `partially-derived`: 140
- `live-observed`: 0
- `not-available`: 55

## Notes

- `vnstock_data_alt` UI/domain layers have richer normalized-schema coverage because they ship explicit schema registries.
- `vnstock_alt` output shape is often only partially derivable unless provider modules publish clear column constants.
- Live snapshots are evidence, not the contract. The source remains the primary contract.
