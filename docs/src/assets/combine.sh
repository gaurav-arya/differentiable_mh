#!/bin/bash

pdfjam rwmh_tuning_full_{A,B,C}.pdf --nup 3x1 --landscape --papersize '{106mm,318mm}' --noautoscale true --outfile rwmh_tuning_full_combined.pdf
pdfjam rwmh_tuning_gaussian_{1,3,5}d.pdf --nup 3x1 --landscape --papersize '{132mm,237mm}' --noautoscale true --outfile rwmh_tuning_gaussian_combined.pdf
