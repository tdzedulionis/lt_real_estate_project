#!/bin/bash

# Download the Microsoft ODBC driver package directly
curl https://packages.microsoft.com/debian/10/prod/pool/main/m/msodbcsql18/msodbcsql18_18.2.1.1-1_amd64.deb -o msodbcsql18.deb

# Install the package (ACCEPT_EULA=Y is required for automated installation)
ACCEPT_EULA=Y dpkg -i msodbcsql18.deb

# Clean up the downloaded package
rm msodbcsql18.deb
