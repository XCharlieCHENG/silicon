#!/usr/bin/env Rscript

# Minimal single-task regression analysis for silicon_demo (Fearspeech)
# - Reads: outputs/Regression/fearspeech_regression.csv
# - Produces: outputs/Plots/regression/combined_model_comparison_<method>.png

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(broom)
  library(lmtest)
  library(sandwich)
  library(ggplot2)
})

# Parse command line argument (kept for compatibility; only 'binary' is used)
args <- commandArgs(trailingOnly = TRUE)
accuracy_method <- if (length(args) > 0) args[1] else "binary"
if (!accuracy_method %in% c("binary")) {
  stop("Only 'binary' accuracy is supported in the demo.")
}

# Input/output paths relative to silicon_demo/
input_file <- file.path("outputs", "Regression", "fearspeech_regression.csv")
plot_path <- file.path("outputs", "Plots", "regression",
                      sprintf("combined_model_comparison_%s.png", accuracy_method))
dir.create(dirname(plot_path), recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input_file)) {
  stop(sprintf("Regression CSV not found: %s", input_file))
}

data <- read.csv(input_file)

# Ensure id column exists
if (!"id" %in% names(data)) {
  data$id <- seq_len(nrow(data))
}

# Detect model columns (all non-id columns)
model_cols <- setdiff(names(data), c("id"))
if (length(model_cols) < 2) {
  stop("Need at least two model columns for comparison.")
}

# Prefer GPT-4o as reference if present
ref_name <- if ("gpt-4o" %in% model_cols) "gpt-4o" else model_cols[[1]]

long_df <- data %>%
  pivot_longer(cols = all_of(model_cols), names_to = "model", values_to = "y") %>%
  mutate(
    model = factor(model),
    model = relevel(model, ref = ref_name)
  )

# Fit logit with clustered SE by id
fit <- glm(y ~ model, data = long_df, family = binomial())
vc <- vcovCL(fit, cluster = ~id, type = "HC1")
coefs <- tidy(fit) %>%
  mutate(
    std.error = sqrt(diag(vc)),
    statistic = estimate / std.error,
    p.value = 2 * pt(-abs(statistic), df = fit$df.residual),
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error
  )

# Plot (exclude intercept; show non-reference models only)
plot_data <- coefs %>%
  filter(grepl("^model", term)) %>%
  mutate(model = sub("^model", "", term))

p <- ggplot(plot_data, aes(x = estimate, y = model)) +
  geom_point(size = 3) +
  geom_errorbar(aes(xmin = conf.low, xmax = conf.high), width = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = sprintf("LLM Model Performance Comparison vs %s (Binary)", ref_name),
    x = "Log Odds Ratio vs reference",
    y = "Model"
  ) +
  theme_minimal()

ggsave(plot_path, p, width = 8, height = 6, dpi = 300)

cat("\n=== Regression Summary (binary) ===\n")
print(plot_data %>% select(model, estimate, std.error, statistic, p.value))
cat(sprintf("\nSaved plot to: %s\n", plot_path))
