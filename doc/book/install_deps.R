#!/usr/bin/env Rscript
install.packages(c("berryFunctions",
                   "gifski",
                   "reticulate",
                   "Rcpp",
                   "RcppEigen",
                   "bookdown",
                   "downlit",
                   "xml2",
                   "servr",
                   "remotes"))

remotes::install_git("https://git.rud.is/hrbrmstr/swiftr.git")
