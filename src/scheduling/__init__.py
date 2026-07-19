"""Scheduling utilities for concurrent multi-GPU committee training.

This package implements the model-level parallelism used by the CCGrid 2027
multi-GPU scaling experiments: N independent committee models are trained
concurrently, one model per GPU, scheduled in waves of size G.
"""
