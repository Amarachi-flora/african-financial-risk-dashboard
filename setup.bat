cat > setup.bat << 'EOF'
@echo off
echo Setting up African Financial Risk Dashboard...
pip install -r requirements.txt
mkdir models outputs charts powerbi eda_reports 2>nul
echo Setup complete!
pause
EOF