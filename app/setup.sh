#!/bin/bash

# Add Microsoft repository
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install ODBC Driver 18
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18
