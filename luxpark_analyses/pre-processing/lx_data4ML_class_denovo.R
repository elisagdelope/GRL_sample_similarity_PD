# Title: lx_data4ML_class_denovo.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script the pre-processing steps prior to ML classification.
# Usage: Rscript lx_data4ML_class_denovo.R 10 
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
analysis_name <- "02-pred-V0-PD"
OUT_DIR <- paste0("../data/", analysis_name , "/02-outfiles") 
target = "DIAGNOSIS" 
myseed = 111
METAB.FILE <- file.path("../data/01-dea-V0-NOVO/02-outfiles/", "log_transformed_V0.tsv")
PHENO.FILE <- file.path("../data/01-dea-V0-NOVO/02-outfiles/", "pheno_V0.tsv")
features_varname <- "METABOLITES_ID"



# Data load --------------------------------------------------------------------
# log-transformed peak area & clinical data.
pheno <- vroom(PHENO.FILE, col_types = c("cccffdiiiddddidii")) 
metab <- vroom(METAB.FILE, col_types = cols()) 
metab = metab[!duplicated(metab[["SAMPLE_ID"]]), ]


# Data reformatting for ML -----------------------------------------------------
# kNN imputation for BMI variable
pheno <- VIM::kNN(pheno, variable = "BMI", k= 5, imp_var = F)

pheno_4ML <- pheno %>%
  dplyr::select(all_of(c(target, 'SAMPLE_ID'))) %>%
  mutate(!!target := case_when(get(target) == "HC" ~ 0,
                               get(target) == "PD" ~ 1))
dim_metab <- dim(metab)


# apply unsupervised filters ---------------------------------------------------
# remove near zero variance features 
nzv = nearZeroVar(metab, names = TRUE)
if (length(nzv) > 0) {
  metab <- metab %>%
    dplyr::select(-any_of(nzv))
}

# remove highly correlated features 
cor_df = cor(metab[,-c(1:3)]) # remove patient ID variables
hc = findCorrelation(cor_df, cutoff=0.85, names = TRUE) 
hc = sort(hc)
if (length(hc) > 0) {
  metab = metab[,-which(names(metab) %in% c(hc))]
}
print("NZV, correlation filters and treatment effects successfully applied")
if (e_level == "METAB") {
  print(paste((length(hc) + length(nzv) + length(mcor) + length(mpw)), "features were removed out of", (dim_metab[2]-3)))
} else {
  print(paste((length(hc) + length(nzv) + length(mcor)), "features were removed out of", (dim_metab[2]-3)))
}


# add target variable information
metab_4ML <- metab %>%
  dplyr::select(-any_of(c("PATIENT_ID", "VISIT"))) %>%
  inner_join(pheno_4ML, 
             by = "SAMPLE_ID") %>%
  mutate_at(target, factor) %>% 
  column_to_rownames("SAMPLE_ID")

rm(pheno_4ML, metab)


# export data
readr::write_tsv(metab_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR, "DENOVO_data_metab_4ML_DIAGNOSIS.tsv"))


# Session info -----------------------------------------------------------------
rm(list = ls())
gc(T)
cat("\n================\n  SESSION INFO\n================\n")
sessionInfo()

