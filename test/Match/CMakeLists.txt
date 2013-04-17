cmake_minimum_required(VERSION 2.6)

project(DOMatch)

find_package(DOMatch REQUIRED)
include(${DOMatch_USE_FILE})

unit_test(DO Match "DOMatch")