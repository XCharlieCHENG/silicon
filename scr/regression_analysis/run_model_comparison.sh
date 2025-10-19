#!/bin/bash

echo ""
echo "Running model comparison..."
Rscript scr/regression_analysis/run_model_comparison.R binary


echo ""
echo "Model comparison plot generated successfully! outputs/Plots/regression/combined_model_comparison_binary.png"
