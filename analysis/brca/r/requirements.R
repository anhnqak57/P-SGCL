# Run manually with: Rscript analysis/brca/r/requirements.R
# Packages live in %LOCALAPPDATA%/PSGCL_test_R/library by default.  This ASCII
# user path avoids Windows R's locale limitation with the repository's Unicode
# parent directory, while still avoiding R's system library. PSGCL_R_LIBS can
# override it for a managed shared R library.
full_arguments <- commandArgs(trailingOnly = FALSE)
script_argument <- grep("^--file=", full_arguments, value = TRUE)
script_path <- if (length(script_argument) == 1L) sub("^--file=", "", script_argument) else "requirements.R"
default_library <- file.path(Sys.getenv("LOCALAPPDATA"), "PSGCL_test_R", "library")
project_library <- Sys.getenv("PSGCL_R_LIBS", unset = default_library)
dir.create(project_library, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(project_library, .libPaths())))

required_packages <- c("jsonlite", "ggplot2", "httr", "curl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org", type = "binary")
}
unavailable_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(unavailable_packages) > 0L) {
  stop("R package installation failed: ", paste(unavailable_packages, collapse = ", "))
}
message("R dependencies available in ", project_library, ": ", paste(required_packages, collapse = ", "))
