#!/bin/bash
# Overnight ETL Run Script
# Runs full 8.6M review ETL with safe memory settings (2g/2g)
# Expected runtime: 5-7 hours on laptop (i7-7500U, 16GB RAM)

set -e  # Exit on error

LOG_FILE="etl_full_run_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "  YELP REVIEW ETL - OVERNIGHT FULL DATASET RUN"
echo "================================================================================"
echo "  Started at:     $(date)"
echo "  Log file:       $LOG_FILE"
echo "  Memory config:  Driver=2g, Executor=2g"
echo "  Dataset:        8.6M reviews (full dataset)"
echo "  Expected time:  5-7 hours"
echo "================================================================================"
echo ""
echo "  System Resources at Start:"
echo "  --------------------------"
free -h | grep -E "Mem:|Swap:" | sed 's/^/  /'
echo ""
df -h /home/jeff | tail -1 | awk '{print "  Disk: " $4 " available"}'
echo ""
echo "================================================================================"
echo ""
echo "  To monitor progress in another terminal, run:"
echo "    ./monitor_etl.sh"
echo ""
echo "  Or tail the log file:"
echo "    tail -f $LOG_FILE"
echo ""
echo "================================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Record start time
START_TIME=$(date +%s)

# Run ETL with full dataset flag and log everything
echo "Starting ETL pipeline..."
python src/1_ETL_Data_Preparation.py --full 2>&1 | tee "$LOG_FILE"

# Capture exit code
ETL_EXIT_CODE=${PIPESTATUS[0]}

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))
SECONDS=$((RUNTIME % 60))

echo ""
echo "================================================================================"
if [ $ETL_EXIT_CODE -eq 0 ]; then
    echo "  ETL RUN COMPLETED SUCCESSFULLY!"
else
    echo "  ETL RUN FAILED! (exit code: $ETL_EXIT_CODE)"
fi
echo "================================================================================"
echo "  Finished at:  $(date)"
echo "  Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Log file:     $LOG_FILE"
echo "================================================================================"

# Show output files
echo ""
echo "  Output files created:"
echo "  ---------------------"
if [ -d "data/processed/01_etl_output" ]; then
    ls -lh data/processed/01_etl_output/ 2>/dev/null | tail -n +2 | sed 's/^/  /'
else
    echo "  (No output directory found)"
fi

# Show system resources after completion
echo ""
echo "  System Resources After Completion:"
echo "  -----------------------------------"
free -h | grep -E "Mem:|Swap:" | sed 's/^/  /'

echo ""
echo "================================================================================"
echo "  Next steps:"
echo "    1. Run: python validate_full_etl.py"
echo "    2. Review $LOG_FILE for any warnings"
echo "    3. If successful, proceed to Stage 2 NLP"
echo "================================================================================"

exit $ETL_EXIT_CODE
