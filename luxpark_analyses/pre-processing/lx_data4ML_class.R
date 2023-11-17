# Title: lx_data4ML_class.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script the pre-processing steps prior to ML classification.
# Usage: Rscript lx_data4ML_class.R 10 
# Data: data from metabolomics at specific timepoint (V0), clinical data (diagnosis).

# GC ---------------------------------------------------------------------------
rm(list = ls())
gc(T)



# Packages ---------------------------------------------------------------------
library(readr)
library(plyr)
library(dplyr)
library(vroom)
library(tidyr)
library(tibble)
library(stringr)
library(caret)
library(limma)
library(argparser, quietly = TRUE)
library(matrixStats)




# I/O --------------------------------------------------------------------------
# cmd line arguments
p <- arg_parser("PreprocessCV", hide.opts = FALSE)
p <- add_argument(parser = p, arg = "--analysis", help = "name of analysis directory (e.g., 02-pred-BL-PD, 02-pred-BL-UPDRS3-class)", required = TRUE)
argv <- parse_args(p, argv = commandArgs(trailingOnly = TRUE))
print(argv)
OUT_DIR <- paste0("../data/", analysis_name , "/02-outfiles") 
OUT_DIR_PATHWAY <- paste0("../data/", analysis_name , "/04-pathway_level") 
OUT_DIR_DATA <- paste0("../data/", analysis_name , "/05-data4ML")
IN_DIR <- "../data/00-cleansing/"
ANNOTATION.FILE <- file.path(IN_DIR, "chemical_annotation.tsv")
M1342.FILE <- file.path(IN_DIR, "M1342.tsv") # M1342 aka 3-methoxytyrosine: confounder
myseed = 111

if (grepl("V0-PD", analysis_name)) {
  PHENO.FILE <- file.path(OUT_DIR, "pheno_V0.tsv")
  target = "DIAGNOSIS"
} else if (grepl("V0-UPDRS3", analysis_name)){
  PHENO.FILE <- file.path(OUT_DIR, "lx_pheno.tsv")
  target = "UPDRS__3_binary"
}
source("func_data4ML_class.R")



# Main -------------------------------------------------------------------------
if ((!dir.exists(OUT_DIR)) | (!dir.exists(OUT_DIR_PATHWAY)) | (!dir.exists(OUT_DIR_DATA)) ) {
  dir.create(OUT_DIR, recursive = T)
  dir.create(OUT_DIR_PATHWAY, recursive = T)
  dir.create(OUT_DIR_DATA, recursive = T)
}


for (e_level in c("METAB", "PW")) {
  if (e_level == "METAB") {
    METAB.FILE <- file.path(OUT_DIR, "log_transformed_V0.tsv")
    features_varname <- "METABOLITES_ID"
    process_data4ML(METAB.FILE, M1342.FILE, PHENO.FILE, features_varname, OUT_DIR_DATA, target, myseed, export=TRUE)
  } else { # aggregationss
    
    for (st in c("mean", "median", "sd", "pca", "pathifier")) {
      METAB.FILE <- file.path(OUT_DIR_PATHWAY, paste0("log_transformed_PW_", st, "_V0.tsv"))
      features_varname <- "PATHWAY_NAME"
      
      if ((!e_level %in% c("PW", "METAB")) | (!st %in% c("mean", "median", "sd", "pathifier", "pca")) | (!file.exists(METAB.FILE))) { 
        stop("Adequate arguments were not provided")
      }
      process_data4ML(METAB.FILE, M1342.FILE, PHENO.FILE, features_varname, OUT_DIR_DATA, target, myseed, export=TRUE)
    }
  }
}    



# Session info -----------------------------------------------------------------
rm(list = ls())
gc(T)
cat("\n================\n  SESSION INFO\n================\n")
sessionInfo()

