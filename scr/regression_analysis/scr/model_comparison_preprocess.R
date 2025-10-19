#' Data Processing Module for LLM Annotation Analysis
#'
#' This module provides functions for loading and preprocessing the regression data
#' for LLM performance comparisons.

library(dplyr)
library(tidyr)
library(readr)

#' Load and preprocess the regression data
#'
#' @param file_path Path to the CSV file containing the regression data
#' @return Processed data frame in long format ready for analysis
load_regression_data <- function(file_path) {
  # Load the data
  data <- read_csv(file_path, show_col_types = FALSE)

  # Generate id variable (equivalent to Stata's gen id = _n)
  data$id <- seq_len(nrow(data))

  # Drop tag_spur_majority column (as in Stata code)
  data <- data %>% select(-tag_spur_majority)

  # Reshape from wide to long format
  # Identify Y columns (treatment variables)
  y_cols <- grep("^Y\\d+$", names(data), value = TRUE)

  # Reshape long by treatment
  data_long <- data %>%
    pivot_longer(
      cols = all_of(y_cols),
      names_to = "treatment_str",
      values_to = "y"
    ) %>%
    mutate(
      # Extract treatment number from column name (remove 'Y' prefix)
      treatment = as.numeric(gsub("Y", "", treatment_str))
    ) %>%
    select(-treatment_str) %>%
    arrange(treatment)

  return(data_long)
}

#' Filter data for model comparison analysis
#'
#' @param data Long-format data frame
#' @return Filtered data frame with selected treatments for model comparison
filter_model_comparison <- function(data) {
  # Treatments for dao cross-model comparison (from Stata line 39)
  model_treatments <- c(4, 16, 8, 9, 38, 39, 57)

  filtered_data <- data %>%
    filter(treatment %in% model_treatments)

  return(filtered_data)
}

#' Filter data for prompt comparison analysis
#'
#' @param data Long-format data frame
#' @return Filtered data frame with selected treatments for prompt comparison
filter_prompt_comparison <- function(data) {
  # Treatments for dao-GPT-4o prompt comparison (from Stata line 65)
  prompt_treatments <- c(16, 18, 19, 20, 27, 28)

  filtered_data <- data %>%
    filter(treatment %in% prompt_treatments)

  return(filtered_data)
}

#' Get treatment labels for plotting
#'
#' @return Named vector of treatment labels
get_treatment_labels <- function() {
  labels <- c(
    "4" = "GPT-4 Turbo",
    "8" = "Gemini 1.5 Pro",
    "9" = "Claude 3.5 Sonnet",
    "16" = "GPT-4o",
    "18" = "System Role; Persona",
    "19" = "System Role; CoT",
    "20" = "User Role; Base",
    "27" = "User Role; Persona",
    "28" = "User Role; CoT",
    "38" = "o3-mini",
    "39" = "LLaMA 3.3 70B",
    "57" = "DeepSeek-R1"
  )
  return(labels)
}


