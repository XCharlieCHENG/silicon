#!/usr/bin/env Rscript
#' Main Analysis Script for LLM Annotation Performance Comparison
#'
#' This script reproduces the Stata analysis in reg_compare.do using modular R code.
#' It compares different LLM models and prompts on annotation performance.

# Load required libraries with error handling
required_packages <- c("dplyr", "readr", "here", "tidyr", "broom", "lmtest", "sandwich", "car", "ggplot2", "stringr")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed. Please run source('requirements.R') first.", pkg))
  }
  library(pkg, character.only = TRUE)
}

# Source the modular components
source("scr/model_comparison_preprocess.R")
source("scr/regression.R")
source("scr/reg_plot.R")

#' Main analysis function
#'
#' @param data_file Path to the CSV data file (default: relative path to output/Regression/dao.csv)
#' @param output_dir Directory to save plots (default: "output/plots")
main_analysis <- function(data_file = "../output/Regression/dao.csv",
                         output_dir = "../output/plots") {

  cat("Starting LLM Annotation Performance Analysis\n")
  cat("==========================================\n\n")

  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Step 1: Load and preprocess data
  cat("1. Loading and preprocessing data...\n")
  data <- load_regression_data(data_file)
  cat(sprintf("   Loaded %d observations with %d unique treatments\n",
              nrow(data), length(unique(data$treatment))))

  # Step 2: Model Comparison Analysis
  cat("\n2. Running model comparison analysis...\n")
  model_data <- filter_model_comparison(data)
  model_results <- run_model_comparison_logit(model_data, reference_treatment = 16)

  # Print model comparison results
  print_model_summary(model_results, "Model Comparison Results (Logit Regression)")

  # Create model comparison plot
  model_plot <- plot_model_comparison(
    model_results,
    file.path(output_dir, "model_comparison.png")
  )
  cat("   Model comparison plot saved to:", file.path(output_dir, "model_comparison.png"), "\n")

  # Step 3: Prompt Comparison Analysis
  cat("\n3. Running prompt comparison analysis...\n")
  prompt_data <- filter_prompt_comparison(data)
  prompt_results <- run_prompt_comparison_lm(prompt_data, reference_treatment = 16)

  # Print prompt comparison results
  print_model_summary(prompt_results, "Prompt Comparison Results (Linear Regression)")

  # Create prompt comparison plot
  prompt_plot <- plot_prompt_comparison(
    prompt_results,
    file.path(output_dir, "prompt_comparison.png")
  )
  cat("   Prompt comparison plot saved to:", file.path(output_dir, "prompt_comparison.png"), "\n")

  # Step 4: Nonparametric Tests
  cat("\n4. Running nonparametric tests...\n")
  nonparametric_results <- run_nonparametric_tests(data)

  # Print nonparametric test results
  cat("\nNonparametric Test Results:\n")
  cat("Kruskal-Wallis test:\n")
  print(nonparametric_results$kruskal_wallis)

  if (!is.null(nonparametric_results$dunn_bonferroni)) {
    cat("\nDunn's test (Bonferroni correction):\n")
    print(nonparametric_results$dunn_bonferroni)
  }

  # Step 5: Create combined plots
  cat("\n5. Creating combined plots...\n")
  combined_plots <- create_combined_plots(
    model_results,
    prompt_results,
    file.path(output_dir, "combined_analysis.png")
  )

  cat("\nAnalysis completed successfully!\n")
  cat("Output files saved to:", output_dir, "\n")

  # Return results for further analysis if needed
  return(list(
    data = data,
    model_results = model_results,
    prompt_results = prompt_results,
    nonparametric_results = nonparametric_results,
    plots = combined_plots
  ))
}

#' Run analysis if script is executed directly
if (!interactive()) {
  # Set working directory to script location
  # When running from command line, use the directory where the script is located
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    # Running in RStudio
    setwd(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    # Running from command line - get script directory from command line args
    args <- commandArgs(trailingOnly = FALSE)
    script_path <- sub("--file=", "", args[grep("--file=", args)])
    if (length(script_path) > 0) {
      setwd(dirname(normalizePath(script_path)))
    }
  }

  # Run main analysis with default parameters
  results <- main_analysis()
}

# Example usage in interactive mode:
# source("model_comparison_analysis.R")
# results <- main_analysis(data_file = "../output/Regression/dao.csv")
#
# Or with custom output directory:
# results <- main_analysis(output_dir = "../output/custom_plots")


