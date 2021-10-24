FROM rocker/verse:4.0.2

MAINTAINER "David OK" <david.ok8@gmail.com>

# To avoid console interaction with apt.
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
COPY . .

RUN apt-get update -y -qq
RUN apt-get install -y -qq python3 python3-pip librsvg2-bin

RUN pip3 install matplotlib numpy sympy

RUN Rscript -e 'devtools::install_cran(c( \
      "berryFunctions", \
      "gifski", \
      "remotes", \
      "reticulate", \
      "Rcpp", "RcppEigen",\
      "tinytex"\
      ))'
RUN Rscript -e 'tinytex::install_tinytex()'

# RUN Rscript -e "bookdown::render_book('index.Rmd', 'all', output_dir = 'public')"
