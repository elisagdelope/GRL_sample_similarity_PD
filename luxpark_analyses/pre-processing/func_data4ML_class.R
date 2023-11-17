# Title: func_data4ML_class.R
# Authorship: Elisa Gomez de Lope, Contact details: elisa.gomezdelope@uni.lu
# Info: This script contains functions to pre-process data prior to ML modelling

process_data4ML <- function(METAB.FILE, M1342.FILE, PHENO.FILE, features_varname, OUT_DIR_DATA, target, myseed, export){
  
  # Data load --------------------------------------------------------------------
  # log-transformed peak area & clinical data.
  annotation <- vroom(ANNOTATION.FILE, col_types = cols())
  M1342 <- vroom(M1342.FILE, col_types = cols()) 
  pheno <- vroom(PHENO.FILE, col_types = cols())
  metab <- vroom(METAB.FILE, col_types = cols()) 
  metab = metab[!duplicated(metab[["SAMPLE_ID"]]), ]
  
  
  
  # Data reformatting for ML -----------------------------------------------------
  
  # kNN imputation for BMI variable
  pheno <- VIM::kNN(pheno, variable = "BMI", k= 5, imp_var = F)
  
  if(!all(pheno[[target]] %in% c(0,1))) {
    pheno_4ML <- pheno %>%
      dplyr::select(all_of(c(target, 'SAMPLE_ID'))) %>%
      mutate(!!target := case_when(get(target) == "HC" ~ 0,
                                   get(target) == "PD" ~ 1))
    pheno <- pheno %>% 
      mutate_at(target, factor)
  } else {
    pheno_4ML <- pheno %>%
      dplyr::select(all_of(c(target, 'SAMPLE_ID')))
    pheno <- pheno %>% 
      mutate_at(target, factor) %>%
      mutate(across(all_of(target), ~factor(.x, labels = make.names(sort(unique(pheno[[target]]))))))
  }
  dim_metab <- dim(metab)
  
  
  
  # apply unsupervised filters ---------------------------------------------------
  # remove near zero variance features 
  nzv = nearZeroVar(metab[,-c(1:3)], names = TRUE)
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
  print(paste((length(hc) + length(nzv)), "features were removed out of", (dim_metab[2]-3)))
  
  
  # add target variable information
  metab_4ML <- metab %>%
    dplyr::select(-any_of(c("PATIENT_ID", "VISIT"))) %>%
    inner_join(pheno_4ML, 
               by = "SAMPLE_ID") %>%
    mutate_at(target, factor) %>% 
    column_to_rownames("SAMPLE_ID")
  
  rm(pheno_4ML, metab)
  
  
  # export pre-processed data 
  if (export == TRUE) {
    if (e_level == "METAB") {
      readr::write_tsv(metab_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_metab_4ML_", target, ".tsv")))
    } else {
      readr::write_tsv(metab_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_metab_4ML_", target, ".tsv")))
    }
  }
  
  
  
  # create training/held-out set -------------------------------------------------
  set.seed(myseed-1)
  inTraining <- createDataPartition(metab_4ML[[as.character(target)]], p = .85, list = FALSE)
  hout_4ML  <- metab_4ML[-inTraining, ]
  metab_4ML <- metab_4ML[inTraining, ] 
  # export pre-processed data split
  if (export == TRUE) {
    if (e_level == "METAB") {
      readr::write_tsv(metab_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_cv_metab_4ML_", target, ".tsv")))
      readr::write_tsv(hout_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0("data_test_metab_4ML_", target, ".tsv")))
    } else {
      readr::write_tsv(metab_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_cv_metab_4ML_", target, ".tsv")))
      readr::write_tsv(hout_4ML %>% rownames_to_column(var = "SAMPLE_ID"), file = file.path(OUT_DIR_DATA, paste0(e_level, "_", st, "_data_test_metab_4ML_", target, ".tsv")))
    }
  }
}