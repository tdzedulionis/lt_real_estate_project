#!/bin/bash
set -e

echo "Installing Microsoft ODBC Driver 18 for SQL Server..."

# Clean up any existing configuration
rm -f /etc/apt/sources.list.d/mssql-release.list
apt-get update
apt-get install -y curl gnupg2

# Add Microsoft repository (modern method)
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install ODBC Driver 18
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Verify installation
echo "Checking ODBC driver installation:"
if [ -f /opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.*.so.* ]; then
  echo "✅ ODBC Driver 18 installed successfully"
  ls -l /opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.*.so.*
else
  echo "❌ ODBC Driver 18 installation failed - driver file not found"
  exit 1
fi