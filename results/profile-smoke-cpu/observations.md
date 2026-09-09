# Profile observations

This is an instrumented run, not a benchmark.

| Operation | Calls | Self CPU time (us) | Self device time (us) |
| --- | ---: | ---: | ---: |
| miniinfer::generation | 3 | 13164.02 | 0.00 |
| VmappedVmappedVmappedVmappedModIndex | 24 | 6266.91 | 0.00 |
| aten::native_layer_norm | 60 | 679.84 | 0.00 |
| aten::tanh | 24 | 364.00 | 0.00 |
| aten::addmm | 96 | 264.80 | 0.00 |
| aten::select | 372 | 240.27 | 0.00 |
| aten::index | 48 | 238.12 | 0.00 |
| aten::_softmax | 24 | 229.42 | 0.00 |
| aten::bmm | 48 | 216.50 | 0.00 |
| aten::_to_copy | 108 | 215.04 | 0.00 |
| aten::view | 600 | 207.13 | 0.00 |
| aten::softmax | 24 | 192.09 | 0.00 |
| aten::add | 132 | 189.96 | 0.00 |
| aten::clone | 24 | 187.25 | 0.00 |
| aten::as_strided | 1194 | 173.60 | 0.00 |

Compare separate benchmark reports before claiming a speedup. Use the trace to inspect allocations and launches. Nsight Compute counters are needed to investigate memory bandwidth and occupancy.
