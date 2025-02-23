#!/bin/bash
OUTPUT_DIR="../data/comparison_tests"
N_TRIALS=10

mkdir -p "$OUTPUT_DIR"

echo "Running comparison tests..."
for seed in $(seq 0 $((N_TRIALS-1))); do
    echo "Trial $seed"
    # Fixed controller
    python3 traj.py --wind --wind-seed $seed
    
    # Adaptive controller (analytical)
    python3 traj.py --wind --adapt --wind-seed $seed
    
    # Adaptive controller (heuristic)
    python3 traj.py --wind --adapt --heuristic --wind-seed $seed
done

# Print summary with correct file patterns
echo -e "\nRESULTS SUMMARY"
echo "=============="
echo -e "\nFIXED CONTROLLER:"
echo "-----------------"
cat "$OUTPUT_DIR"/results__normal_wind_full.txt

echo -e "\nADAPTIVE CONTROLLER:"
echo "-------------------"
cat "$OUTPUT_DIR"/results__adaptive_wind_full.txt

echo -e "\nHEURISTIC CONTROLLER:"
echo "--------------------"
cat "$OUTPUT_DIR"/results__adaptive_heuristic_wind_full.txt

# Print a comparison table
echo -e "\nCOMPARISON TABLE"
echo "==============="
printf "%-20s %-15s %-15s %-15s\n" "Controller" "Iterations" "Traj Cost" "Avg Violation"
echo "------------------------------------------------------------"
for type in "normal" "adaptive" "adaptive_heuristic"; do
    iters=$(grep "iterations:" "$OUTPUT_DIR/results__${type}_wind_full.txt" | cut -d' ' -f2)
    cost=$(grep "traj_cost:" "$OUTPUT_DIR/results__${type}_wind_full.txt" | cut -d' ' -f2)
    viol=$(grep "avg_violation:" "$OUTPUT_DIR/results__${type}_wind_full.txt" | cut -d' ' -f2)
    case $type in
        "normal") name="Fixed" ;;
        "adaptive") name="Adaptive" ;;
        "adaptive_heuristic") name="Heuristic" ;;
    esac
    printf "%-20s %-15.2f %-15.4f %-15.6f\n" "$name" "$iters" "$cost" "$viol"
done