# Deconvolve bulk expression into the quantities the twin's observation
# operator reports (tme_twin/observe.py FEATURES).
#
#   Rscript deconvolve.R <in.csv> <platform> <out.csv>
#
# in.csv    rows = probes or genes, first column the identifier, remaining
#           columns = samples. LINEAR scale, not log. The Python driver is
#           responsible for undoing log2 before calling this; see the scale
#           check there. EPIC expects TPM-like values and will return
#           plausible but wrong fractions if given log data.
# platform  affy_hgu133a | illumina_v3 | ensembl
#
# Cell fractions come from EPIC (Racle 2017) with the TRef reference, which
# covers CAFs, endothelium, macrophages and CD8 T cells; "otherCells" is read
# as tumour. EPIC gives one fibroblast-lineage compartment and no M1/M2 split,
# so m2_share and caf_share are taken from ssGSEA scores on polarisation and
# activation signatures instead -- the twin's observe.py treats both as
# monotone-linked scores rather than true fractions for exactly this reason.

suppressPackageStartupMessages({
  library(EPIC); library(GSVA); library(msigdbr)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
infile <- args[1]; platform <- args[2]; outfile <- args[3]

message("reading ", infile)
x <- read.csv(infile, row.names = 1, check.names = FALSE)
x <- as.matrix(x)
message("  ", nrow(x), " features x ", ncol(x), " samples; ",
        "range ", signif(min(x, na.rm = TRUE), 3), " to ", signif(max(x, na.rm = TRUE), 3))
if (max(x, na.rm = TRUE) < 100) {
  stop("values look log-scaled (max < 100). EPIC needs linear TPM-like input; ",
       "convert before calling this script.")
}

# ---- identifiers to gene symbols -------------------------------------------
to_symbol <- function(ids, platform) {
  if (platform == "affy_hgu133a") {
    suppressPackageStartupMessages(library(hgu133a.db))
    unlist(mapply(function(i) {
      s <- AnnotationDbi::mapIds(hgu133a.db, i, "SYMBOL", "PROBEID",
                                 multiVals = "first"); s }, list(ids)))
  } else if (platform == "illumina_v3") {
    suppressPackageStartupMessages(library(illuminaHumanv3.db))
    AnnotationDbi::mapIds(illuminaHumanv3.db, ids, "SYMBOL", "PROBEID",
                          multiVals = "first")
  } else if (platform == "ensembl") {
    suppressPackageStartupMessages(library(org.Hs.eg.db))
    AnnotationDbi::mapIds(org.Hs.eg.db, sub("\\..*$", "", ids), "SYMBOL",
                          "ENSEMBL", multiVals = "first")
  } else stop("unknown platform: ", platform)
}

sym <- to_symbol(rownames(x), platform)
keep <- !is.na(sym)
message("  mapped ", sum(keep), " / ", length(sym), " identifiers to symbols")
x <- x[keep, , drop = FALSE]; sym <- sym[keep]

# Several probes per gene: keep the one with the highest mean, the usual
# convention and the one EPIC's own examples use.
ord <- order(rowMeans(x, na.rm = TRUE), decreasing = TRUE)
x <- x[ord, , drop = FALSE]; sym <- sym[ord]
dup <- duplicated(sym)
x <- x[!dup, , drop = FALSE]
rownames(x) <- sym[!dup]
message("  collapsed to ", nrow(x), " genes")

# ---- EPIC ------------------------------------------------------------------
message("running EPIC (TRef)")
ep <- EPIC::EPIC(bulk = x, reference = "TRef")
frac <- as.data.frame(ep$cellFractions)
message("  EPIC columns: ", paste(colnames(frac), collapse = ", "))

# ---- ssGSEA ----------------------------------------------------------------
hm <- msigdbr(species = "Homo sapiens", collection = "H")
gs <- split(hm$gene_symbol, hm$gs_name)
want <- c("HALLMARK_HYPOXIA", "HALLMARK_TGF_BETA_SIGNALING",
          "HALLMARK_IL6_JAK_STAT3_SIGNALING")
missing <- setdiff(want, names(gs))
if (length(missing)) stop("hallmark sets not found: ", paste(missing, collapse = ", "))
sets <- gs[want]

# M2 polarisation and CAF activation markers. Kept short and explicit so the
# paper can list them; both feed features the twin treats as monotone-linked
# scores, not fractions.
sets[["M2_POLARISATION"]] <- c("CD163", "MRC1", "MSR1", "IL10", "CCL18",
                               "CD200R1", "TGM2", "CCL22", "CLEC7A")
sets[["CAF_ACTIVATION"]]  <- c("ACTA2", "FAP", "POSTN", "COL11A1", "TAGLN",
                               "THY1", "PDGFRB", "COL1A1", "INHBA")

message("running ssGSEA on ", length(sets), " sets")
par <- GSVA::ssgseaParam(x, sets)
sc <- t(GSVA::gsva(par))
message("  scores: ", paste(colnames(sc), collapse = ", "))

out <- cbind(sample = rownames(frac), frac, as.data.frame(sc)[rownames(frac), , drop = FALSE])
write.csv(out, outfile, row.names = FALSE)
message("wrote ", outfile, " (", nrow(out), " samples)")
