# SAIF Dumper

A Rust library with Python bindings for creating SAIF (Switching Activity Interchange Format) files.

## Features

- Load data from Python dictionaries
- Create SAIF files with proper header information
- Generate SAIF net entries with switching activity data
- Support for numpy arrays for efficient data handling

## Usage

### Python Interface

```python
import numpy as np
import saif_dumper

# Create a SAIF dumper instance
dumper = saif_dumper.PySaifDumper()

# Load data from a Python dictionary
# The dictionary keys should be string representations of indices
# The values should be 2-element arrays where the second element is the net name
data = {
    "0": ["signal1", "clk"],
    "1": ["signal2", "data_in"],
    "2": ["signal3", "data_out"]
}
dumper.load_from_dict(data)

# Create arrays for switching activity data (can be numpy arrays or regular Python lists)
tc_master = [150, 200, 100]  # Transition counts (uint32 values)
t0s_master = [50, 100, 25]   # T0 values (uint64 values)

# Create SAIF file
dumper.create_saif_file(
    block_name="test_block",
    test_name="test_name", 
    test_duration=1000,  # Duration in picoseconds
    instance_name="test_instance",
    tc_master=tc_master,
    t0s_master=t0s_master
)
```

### Rust Interface

```rust
use saif_dumper::SaifDumper;
use std::collections::HashMap;
use serde_json::json;

let mut dumper = SaifDumper::new();

// Load data
let mut dict = HashMap::new();
dict.insert("0".to_string(), json!(["signal1", "clk"]));
dict.insert("1".to_string(), json!(["signal2", "data_in"]));
dumper.load_from_dict(dict);

// Create arrays
let tc_master = vec![150u32, 200u32];
let t0s_master = vec![50u64, 100u64];

// Create SAIF file
dumper.create_saif_file(
    "test_block", 
    "test_name", 
    1000, 
    "test_instance", 
    &tc_master, 
    &t0s_master
)?;
```

## SAIF File Format

The generated SAIF file includes:

1. **Header Information**: SAIF version, direction, design info, date, vendor, etc.
2. **Net Entries**: For each net, the file includes:
   - Net name
   - T0 (time when net becomes active)
   - T1 (time when net becomes inactive = duration - T0)
   - TX (transition time, set to 0)
   - TC (transition count)
   - IG (ignore flag, set to 0)

### Example SAIF Output

```
(SAIFILE
(SAIFVERSION "2.0")
(DIRECTION "backward")
(DESIGN )
(DATE "2024-01-15 10:30:00")
(VENDOR "GATSPI Open Source")
(PROGRAM_NAME "GATSPI")
(VERSION "Open Source")
(DIVIDER / )
(TIMESCALE 1 ps)
(DURATION 1000)
(INSTANCE test_instance
  (INSTANCE dut
    (NET
      (clk
        (T0 50) (T1 950) (TX 0)
        (TC 150) (IG 0)
      )
      (data_in
        (T0 100) (T1 900) (TX 0)
        (TC 200) (IG 0)
      )
    )
  )
)
```


