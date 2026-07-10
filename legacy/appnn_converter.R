#!/usr/bin/env Rscript
# APPNN Parser&Converter
# Usage: Rscript appnn_converter.R <input.fasta>

# ==================== PACKAGE MANAGEMENT ====================

install_required_packages <- function() {
  required_packages <- c("dplyr", "tidyr", "readr", "stringr", "purrr")
  

  installed_packages <- rownames(installed.packages())
  missing_packages <- setdiff(required_packages, installed_packages)
  
  if (length(missing_packages) > 0) {
    cat("Installing required packages:", paste(missing_packages, collapse = ", "), "\n")
    
    # Install from CRAN
    tryCatch({
      install.packages(missing_packages, repos = "https://cloud.r-project.org")
      cat("Successfully installed packages:", paste(missing_packages, collapse = ", "), "\n")
    }, error = function(e) {
      cat("Error installing packages:", e$message, "\n")
      cat("Please install the following packages manually:\n")
      cat(paste("  install.packages(c('", paste(missing_packages, collapse = "', '"), "'))\n", sep = ""))
      quit(status = 1)
    })
  }
}

check_appnn_package <- function() {
  if (!requireNamespace("appnn", quietly = TRUE)) {
    cat("The 'appnn' package is not installed.\n")
    cat("Please install the appnn package manually before running this script.\n")
    quit(status = 1)
  }
}


install_required_packages()
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
})

# Configuration
INTERMEDIATE_CSV <- "output_proteins.csv"
OUTPUT_DIR <- "APPNN_parsed"

# ==================== UTILITY FUNCTIONS ====================

#' Parse command line arguments
parse_arguments <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript parser_appnn.R <input.fasta>\n")
    cat("Example: Rscript parser_appnn.R proteins.fasta\n")
    quit(status = 1)
  }
  
  list(input_fasta = args[1])
}

#' Read and parse FASTA file
read_fasta_file <- function(file_path) {
  cat("Reading FASTA file:", file_path, "\n")
  
  lines <- readLines(file_path, warn = FALSE) %>% 
    str_trim() %>% 
    keep(~ nchar(.x) > 0)
  
  if (length(lines) == 0) {
    stop("FASTA file is empty or cannot be read.")
  }
  
  sequences <- character()
  names <- character()
  current_seq <- character()
  current_name <- character()
  seq_count <- 0L
  
  for (line in lines) {
    if (str_starts(line, ">")) {
      # Save previous sequence if exists
      if (length(current_seq) > 0) {
        seq_count <- seq_count + 1L
        sequences[[seq_count]] <- paste(current_seq, collapse = "")
        names[[seq_count]] <- current_name
      }
      # Start new sequence
      current_name <- str_sub(line, 2)
      current_seq <- character()
    } else {
      # Append to current sequence
      clean_line <- str_remove_all(line, "[[:space:][:digit:]]")
      current_seq <- c(current_seq, clean_line)
    }
  }
  
  # Save the last sequence
  if (length(current_seq) > 0) {
    seq_count <- seq_count + 1L
    sequences[[seq_count]] <- paste(current_seq, collapse = "")
    names[[seq_count]] <- current_name
  }
  
  tibble(name = unlist(names, use.names = FALSE), sequence = unlist(sequences, use.names = FALSE))
}

#' Format hotspot regions as string
format_hotspots <- function(hotspots) {
  if (is.null(hotspots) || length(hotspots) == 0) {
    return(NA_character_)
  }
  
  if (is.list(hotspots)) {
    hotspot_str <- map_chr(hotspots, ~ paste(.x, collapse = "-"))
    return(paste(hotspot_str, collapse = ";"))
  } else if (is.matrix(hotspots)) {
    return(toString(as.vector(hotspots)))
  } else {
    return(toString(hotspots))
  }
}

# ==================== STEP 1: APPNN ANALYSIS ====================

run_appnn_analysis <- function(fasta_data) {
  cat("Running APPNN analysis...\n")
  
  # Check if appnn package is available
  check_appnn_package()
  
  sequences <- fasta_data$sequence
  predictions <- appnn::appnn(sequences)
  
  cat("Found", length(predictions), "predictions\n")
  
  # Process predictions into a tidy format
  results <- map_dfr(seq_along(predictions), function(i) {
    pred <- predictions[[i]]
    
    tibble(
      sequence_name = fasta_data$name[i],
      sequence = fasta_data$sequence[i],
      overall = if (length(pred) >= 2) toString(pred[[2]]) else NA_character_,
      aminoacids = if (length(pred) >= 3) {
        aa_scores <- pred[[3]]
        if (is.matrix(aa_scores)) {
          toString(as.vector(aa_scores))
        } else {
          toString(aa_scores)
        }
      } else NA_character_,
      hotspots = if (length(pred) >= 4) {
        format_hotspots(pred[[4]])
      } else NA_character_
    )
  })
  
  return(results)
}

save_intermediate_results <- function(results, output_file) {
  cat("Saving intermediate results to:", output_file, "\n")
  write_csv(results, output_file)
  results
}

# ==================== STEP 2: PARSE AND SAVE RESULTS ====================

parse_protein_data <- function(results_df) {
  cat("Parsing protein data...\n")
  
  # Create a list to store individual protein data frames
  protein_data_list <- list()
  
  for (i in seq_len(nrow(results_df))) {
    row <- results_df[i, ]
    
    # Split sequence into amino acids
    aminoacids <- str_split(row$sequence, "", simplify = FALSE)[[1]]
    
    # Parse amino acid scores
    aa_scores_str <- row$aminoacids
    if (!is.na(aa_scores_str)) {
      # Clean the string and convert to numeric
      aa_scores <- aa_scores_str %>% 
        str_remove_all("[\\[\\]]") %>% 
        str_split(",\\s*") %>% 
        .[[1]] %>% 
        as.numeric()
    } else {
      aa_scores <- rep(NA_real_, length(aminoacids))
    }
    
    # Parse hotspot regions
    hotspot_positions <- numeric()
    hotspots_str <- row$hotspots
    
    if (!is.na(hotspots_str) && hotspots_str != "") {
      ranges <- str_split(hotspots_str, ";")[[1]]
      
      for (range_str in ranges) {
        if (str_detect(range_str, "-")) {
          positions <- str_split(range_str, "-")[[1]] %>% 
            as.numeric()
          if (length(positions) == 2) {
            hotspot_positions <- c(hotspot_positions, seq(positions[1], positions[2]))
          }
        }
      }
    }
    
    # Create tidy data frame for this protein
    protein_df <- tibble(
      sequence_name = row$sequence_name,
      aminoacid_position = seq_along(aminoacids),
      aminoacid = aminoacids,
      aminoacid_score = aa_scores[seq_along(aminoacids)],
      hotspot_region = as.integer(seq_along(aminoacids) %in% hotspot_positions)
    )
    
    # Store in list with sequence name as key
    protein_data_list[[row$sequence_name]] <- protein_df
  }
  
  # Combine all proteins
  bind_rows(protein_data_list, .id = "original_name")
}

save_individual_proteins <- function(parsed_data, output_dir) {
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat("Created output directory:", output_dir, "\n")
  }
  
  # Split by protein and save individual files
  parsed_data %>% 
    group_by(sequence_name) %>% 
    group_walk(~ {
      # Clean filename
      clean_name <- str_replace_all(.y$sequence_name, "[^[:alnum:]_]", "_")
      filename <- paste0(clean_name, "_APPNN.csv")
      filepath <- file.path(output_dir, filename)
      
      # Remove the grouping column before saving
      write_csv(.x %>% select(-original_name), filepath)
      cat("Saved:", filepath, "\n")
    })
}

# ==================== MAIN PIPELINE ====================

main <- function() {
  cat("========================================\n")
  cat("APPNN Output Parser\n")
  cat("========================================\n")
  
  # Step 0: Parse arguments
  args <- parse_arguments()
  input_fasta <- args$input_fasta
  
  cat("Input file:", input_fasta, "\n")
  
  # Step 1: Read FASTA
  fasta_data <- tryCatch(
    {
      data <- read_fasta_file(input_fasta)
      cat("Found", nrow(data), "protein sequences\n")
      data
    },
    error = function(e) {
      cat("Error reading FASTA file:", e$message, "\n")
      quit(status = 1)
    }
  )
  
  # Step 2: Run APPNN analysis
  intermediate_results <- tryCatch(
    {
      results <- run_appnn_analysis(fasta_data)
      save_intermediate_results(results, INTERMEDIATE_CSV)
    },
    error = function(e) {
      cat("Error in APPNN analysis:", e$message, "\n")
      # Create empty results if APPNN fails
      tibble(
        sequence_name = fasta_data$name,
        sequence = fasta_data$sequence,
        overall = NA_character_,
        aminoacids = NA_character_,
        hotspots = NA_character_
      )
    }
  )
  
  # Step 3: Parse and save results
  tryCatch(
    {
      parsed_data <- parse_protein_data(intermediate_results)
      save_individual_proteins(parsed_data, OUTPUT_DIR)
      
      # Clean up intermediate file
      if (file.exists(INTERMEDIATE_CSV)) {
        file.remove(INTERMEDIATE_CSV)
        cat("Intermediate files deleted\n")
      }
      
      # Final summary
      cat("========================================\n")
      cat("Parsing completed successfully!\n")
      cat("Output directory:", OUTPUT_DIR, "\n")
      cat("Proteins processed:", n_distinct(parsed_data$sequence_name), "\n")
      cat("========================================\n")
      invisible(TRUE)
    },
    error = function(e) {
      cat("Error in parsing/saving results:", e$message, "\n")
      quit(status = 1)
    }
  )
}

# Run the pipeline
if (!interactive()) {
  main()
}
