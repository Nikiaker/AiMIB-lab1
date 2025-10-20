# Tworzenie plików .sim z różnymi siłami mutacji
echo "Creating mutation configuration files..."
for M in "${MUTATION_RATES[@]}"; do
    VALUE=$(echo "scale=2; $M / 100" | bc -l)
    
    cat > "f9-mut-$M.sim" << EOF
sim_params:
f9_mut:$VALUE
EOF
    echo "Created f9-mut-$M.sim with f9_mut=$VALUE"
done