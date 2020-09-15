# Data preprocessing: This script will replace all indefinite blanks of data with NA

library(mice)
library(dplyr)

# Original file address
org_file <- "C://Users/lihanmin/Desktop/data_processing/data_of_prognosis.CSV"

# Load data
org_data <- read.table(org_file, header = F, check.names = F, sep = ",", na.strings = c("Not Found", "", "<1.00", "Î´²â³ö"))

# Processing
deleted_row <- c(1, 49, 50, 51, 52, 53, 54)
org_data <- org_data[-deleted_row, ]

# Write into CSV
target_file <- "C://Users/lihanmin/Desktop/data_processing/raw_data.CSV"
write.table(org_data, target_file, row.names = FALSE, col.names = FALSE, sep = ",")