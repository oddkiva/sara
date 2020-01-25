#!/bin/bash

# Add the DEB repository.
sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt bionic-pgdg main" >> /etc/apt/sources.list'
wget --quiet -O - http://apt.postgresql.org/pub/repos/apt/ACCC4CF8.asc | apt-key add -
apt update -y

# Install postgresql and postgis.
apt install -y \
  postgresql-11 \
  postgresql-11-postgis-2.5 \
  postgresql-11-postgis-2.5-scripts \
  postgresql-11-pgrouting


sudo -u postgres createuser david
sudo -u postgres createdb gisdb
psql gisdb

# Then:
#  $psql
#  gisdb=#create database gisdb
#  gisdb=#create user david with encrypted password 'mypass';
#  gisdb=#grant all privileges on database gisdb to david;
