cat > setup.sh << 'EOF'
#!/bin/bash
echo "Setting up African Financial Risk Dashboard..."
pip install -r requirements.txt
mkdir -p models outputs charts powerbi eda_reports
echo "Setup complete!"
EOF
