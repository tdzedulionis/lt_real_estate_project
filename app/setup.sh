#!/bin/bash

# Add Microsoft repository signing key
curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc

# Add Microsoft repository
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Update package list
apt-get update

# Install SQL Server ODBC Driver (automatically accepts EULA)
ACCEPT_EULA=Y apt-get install -y msodbcsql18
