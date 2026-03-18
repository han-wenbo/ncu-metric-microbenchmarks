Compile the cuda codes:
```
nvcc -O3 -std=c++17 -lineinfo  <NAME>.cu -o <NAME>
```

Profiling them:
```
ncu \
  --import-source on \
  -f \
  --section SourceCounters \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --metrics \
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_read.sum.peak_sustained,\
lts__t_sectors_op_read.avg,\
lts__t_sectors_op_read.avg.peak_sustained,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sector_op_read_hit_rate.pct,\
lts__t_sectors_op_read_lookup_miss.sum \
  --source-folders  <PATH/TO/THE/MIRCOBENCHMARK/SOURCE/FILE> \
  -o report \
  <PATH/TO/OUTPUT/REPORT>
```

After that, you can use the GUI NCU to open the output reports.
