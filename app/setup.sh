#!/bin/bash

# Install SSL dependencies
apt-get update && apt-get install -y openssl

# Download the Microsoft ODBC driver package directly
curl https://packages.microsoft.com/debian/10/prod/pool/main/m/msodbcsql18/msodbcsql18_18.2.1.1-1_amd64.deb -o msodbcsql18.deb

# Install the package (ACCEPT_EULA=Y is required for automated installation)
ACCEPT_EULA=Y dpkg -i msodbcsql18.deb

# Configure ODBC driver with proper SSL settings
odbcinst -i -d -f /opt/microsoft/msodbcsql18/etc/odbcinst.ini

# Clean up downloaded package
rm msodbcsql18.deb
