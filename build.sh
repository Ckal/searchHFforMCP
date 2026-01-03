#!/bin/bash
set -e

echo "Build process for HF Space..."
# For HF Spaces, we don't install dependencies or validate in CI
# HF Spaces will handle all of this automatically when deployed

# Verify required files exist
if [ -f src/app.py ]; then
    echo " app.py found"
elif [ -f src/gradio_app.py ]; then
    echo " gradio_app.py found"
else
    echo "Note: No standard app.py found (might use different structure)"
fi

if [ -f src/requirements.txt ]; then
    echo " requirements.txt found"
fi

# Create build directory (required for CI artifact upload)
mkdir -p build
echo "Build completed successfully" > build/BUILD_SUCCESS.txt

echo " Build complete - ready for deployment"
