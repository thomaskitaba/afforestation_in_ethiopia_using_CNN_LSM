#!/bin/bash

# Create streamlit configuration directory
mkdir -p ~/.streamlit/

# Create credentials file
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

# Create config file
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml

# Install any additional system dependencies if needed
# apt-get update && apt-get install -y any-packages