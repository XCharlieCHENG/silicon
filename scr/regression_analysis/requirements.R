# Requirements file for LLM Annotation Analysis
#
# Run this script to install all required packages:
# source("requirements.R")

# Set CRAN mirror to avoid mirror selection issues
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# List of required packages
required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "broom",
  "lmtest",
  "sandwich",
  "car",
  "ggplot2",
  "stringr",
  "FSA",
  "rstudioapi",
  "here"
)

# Install missing packages with better error handling
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing package: %s\n", pkg))
    tryCatch({
      install.packages(pkg, quiet = TRUE, dependencies = TRUE)
      cat(sprintf("Successfully installed: %s\n", pkg))
    }, error = function(e) {
      cat(sprintf("Failed to install %s: %s\n", pkg, e$message))
      cat(sprintf("You may need to install %s manually.\n", pkg))
    })
  } else {
    cat(sprintf("Package already installed: %s\n", pkg))
  }
}

cat("Checking and installing required packages...\n")
cat("===========================================\n")

# Install all required packages
results <- lapply(required_packages, install_if_missing)

cat("\nPackage installation check complete!\n")
cat("You can now run: source('main_analysis.R')\n")
