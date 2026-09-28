#!/usr/bin/env Rscript
# Score peptides with AmyloGram and write id,probability as CSV.
# Deliberately minimal: the windowing and the projection back onto residues are
# done in Python, where they are testable and where the modelling choice is
# visible, rather than being buried in an R helper.
suppressMessages({
  library(AmyloGram)
  library(seqinr)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: amylogram_predict.R <peptides.fasta> <out.csv>")
fasta_path <- args[[1]]
out_path   <- args[[2]]

seqs <- seqinr::read.fasta(fasta_path, seqtype = "AA", as.string = FALSE)
data(AmyloGram_model, package = "AmyloGram", envir = environment())
pred <- predict(AmyloGram_model, seqs)

prob <- if (is.data.frame(pred)) {
  col <- intersect(c("Probability", "probability", "prob"), colnames(pred))
  if (length(col) == 0) pred[[ncol(pred)]] else pred[[col[[1]]]]
} else as.numeric(pred)

write.csv(
  data.frame(name = names(seqs), probability = prob, stringsAsFactors = FALSE),
  out_path, row.names = FALSE, quote = FALSE
)