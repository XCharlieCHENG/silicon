#' Model Comparison Module for LLM Annotation Analysis
#'
#' This module provides functions for running regression analyses to compare
#' different LLM models using binary accuracy data.

library(dplyr)
library(broom)
library(lmtest)
library(sandwich)
library(car)
library(ggplot2)
library(tidyr)

# Define model names corresponding to Y1-Y7
MODEL_NAMES <- c(
  "GPT-4o",           # Y1 (reference)
  "Gemini 1.5 Pro",   # Y2
  "LLaMA 3.3 70B",    # Y3
  "Claude 3.5 Sonnet", # Y4
  "o3-mini",          # Y5
  "DeepSeek-R1",      # Y6
  "GPT-OSS 120B"      # Y7
)

#' Convert wide format regression data to long format for analysis
#'
#' @param data Wide format data frame with Y1-Y7 columns and id
#' @return Long format data frame with treatment and y columns
prepare_regression_data <- function(data) {
  data_long <- data %>%
    pivot_longer(
      cols = starts_with("Y"),
      names_to = "treatment",
      values_to = "y",
      names_prefix = "Y"
    ) %>%
    mutate(
      treatment = as.integer(treatment),
      treatment_name = factor(MODEL_NAMES[treatment], levels = MODEL_NAMES)
    )

  return(data_long)
}

#' Run logit regression for model comparison
#'
#' @param data Long format data frame for model comparison
#' @param reference_treatment Reference treatment level (default 1 for GPT-4o)
#' @return List containing model results and F-test results
run_model_comparison_logit <- function(data, reference_treatment = 1) {
  # Ensure treatment is factor with correct reference
  data$treatment <- factor(data$treatment)
  data$treatment <- relevel(data$treatment, ref = as.character(reference_treatment))

  # Run logit regression with clustered standard errors
  model <- glm(y ~ treatment, data = data, family = binomial())

  # Get robust standard errors clustered by id
  robust_se <- vcovCL(model, cluster = ~id, type = "HC1")

  # Extract coefficients and standard errors
  coef_results <- tidy(model) %>%
    mutate(
      std.error = sqrt(diag(robust_se)),
      statistic = estimate / std.error,
      p.value = 2 * pt(-abs(statistic), df = model$df.residual),
      conf.low = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error
    )

  # F-test for treatment effects (handle potential singularity)
  f_test <- tryCatch({
    linearHypothesis(model, grep("treatment", names(coef(model)), value = TRUE),
                     vcov = robust_se, test = "F")
  }, error = function(e) {
    # Return a placeholder result if F-test fails
    list(F = c(NA, NA), Df = c(NA, NA), `Pr(>F)` = c(NA, NA))
  })

  return(list(
    model = model,
    coefficients = coef_results,
    f_test = f_test,
    robust_vcov = robust_se
  ))
}

#' Create coefficient plot for model comparison
#'
#' @param results Model results from run_model_comparison_logit
#' @param title Plot title
#' @param save_path Optional path to save plot
#' @return ggplot object
create_coefficient_plot <- function(results, title = "Model Comparison", save_path = NULL) {
  # Filter to treatment coefficients only
  coef_data <- results$coefficients %>%
    filter(grepl("^treatment", term)) %>%
    mutate(
      model = MODEL_NAMES[as.integer(gsub("treatment", "", term))],
      model = factor(model, levels = MODEL_NAMES[-1])  # Exclude reference
    )

  # Check for models with invalid confidence intervals (constant predictors)
  invalid_ci <- coef_data %>%
    filter(is.na(conf.low) | is.na(conf.high) | is.infinite(conf.low) | is.infinite(conf.high))

  if (nrow(invalid_ci) > 0) {
    warning(sprintf("Models with invalid confidence intervals (likely due to constant predictors): %s",
                    paste(invalid_ci$model, collapse = ", ")))
  }

  # Create plot
  p <- ggplot(coef_data, aes(x = estimate, y = model)) +
    geom_point(size = 3) +
    # Only add error bars for models with valid confidence intervals
    geom_errorbar(data = coef_data %>% filter(!is.na(conf.low) & !is.na(conf.high) &
                                               !is.infinite(conf.low) & !is.infinite(conf.high)),
                  aes(xmin = conf.low, xmax = conf.high), width = 0.2, orientation = "y") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = title,
      x = "Log Odds Ratio vs GPT-4o",
      y = "Model",
      caption = if(nrow(invalid_ci) > 0) paste("Note: No confidence intervals for models with constant predictions:",
                                              paste(invalid_ci$model, collapse = ", ")) else NULL
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.y = element_text(size = 10),
      plot.caption = element_text(size = 8, hjust = 0)
    )

  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 8, height = 6, dpi = 300)
  }

  return(p)
}

#' Print model summary
#'
#' @param results Model results list from run_model_comparison_logit
#' @param title Title for the output
print_model_summary <- function(results, title = "Model Results") {
  cat("\n", title, "\n")
  cat(rep("=", nchar(title)), "\n")

  # Print F-test results (handle potential failure)
  cat("\nF-test for treatment effects:\n")
  if (!is.na(results$f_test$F[2])) {
    cat("F-statistic:", results$f_test$F[2], "\n")
    cat("Degrees of freedom:", results$f_test$Df[2], "\n")
    cat("P-value:", results$f_test$`Pr(>F)`[2], "\n")
  } else {
    cat("F-test could not be computed (possibly due to singularity)\n")
  }

  # Print coefficient estimates
  cat("\nCoefficient estimates (with robust standard errors):\n")
  results$coefficients %>%
    filter(grepl("^treatment", term)) %>%
    mutate(model = MODEL_NAMES[as.integer(gsub("treatment", "", term))]) %>%
    select(model, estimate, std.error, statistic, p.value) %>%
    print()
}


