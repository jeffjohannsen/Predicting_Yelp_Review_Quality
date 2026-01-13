#!/bin/bash
# ETL Monitoring Script
# Run this in another terminal to monitor overnight ETL progress

echo "================================================================================"
echo "  ETL PROGRESS MONITOR"
echo "================================================================================"
echo "  Press Ctrl+C to exit"
echo ""

# Find the most recent log file
find_log_file() {
    ls -t etl_full_run_*.log 2>/dev/null | head -1
}

while true; do
    clear
    echo "================================================================================"
    echo "  ETL PROGRESS MONITOR - $(date)"
    echo "================================================================================"
    echo ""

    # Check if ETL is running
    if pgrep -f "1_ETL_Data_Preparation.py" > /dev/null; then
        echo "  Status: ✓ ETL RUNNING"

        # Get process info
        PID=$(pgrep -f "1_ETL_Data_Preparation.py" | head -1)
        if [ -n "$PID" ]; then
            ELAPSED=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
            echo "  PID:    $PID"
            echo "  Uptime: $ELAPSED"
        fi
    else
        echo "  Status: ✗ ETL NOT RUNNING"
    fi

    echo ""
    echo "  System Resources:"
    echo "  -----------------"
    free -h | grep -E "Mem:|Swap:" | sed 's/^/  /'

    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "  CPU:   ${CPU_USAGE}% used"

    echo ""
    echo "  Current Stage (from log):"
    echo "  -------------------------"
    LOG_FILE=$(find_log_file)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "  Log: $LOG_FILE"
        echo ""

        # Find current stage
        CURRENT_STAGE=$(grep -E "^.*STAGE [0-9]+/[0-9]+:" "$LOG_FILE" | tail -1)
        if [ -n "$CURRENT_STAGE" ]; then
            echo "  $CURRENT_STAGE"
        fi

        # Count completed stages
        COMPLETED=$(grep -c "STAGE.*COMPLETE" "$LOG_FILE" 2>/dev/null || echo "0")
        echo ""
        echo "  Stages completed: $COMPLETED/8"

        # Show progress bar
        PROGRESS=$((COMPLETED * 100 / 8))
        FILLED=$((PROGRESS / 5))
        EMPTY=$((20 - FILLED))
        printf "  Progress: ["
        printf "%${FILLED}s" | tr ' ' '='
        printf "%${EMPTY}s" | tr ' ' '-'
        printf "] %d%%\n" $PROGRESS

        echo ""
        echo "  Recent Activity (last 8 lines):"
        echo "  --------------------------------"
        tail -n 8 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
    else
        echo "  (No log file found - ETL may not have started yet)"
    fi

    echo ""
    echo "  Output Files:"
    echo "  -------------"
    if [ -d "data/processed/01_etl_output" ]; then
        OUTPUT_COUNT=$(find data/processed/01_etl_output -name "*.parquet" -type d 2>/dev/null | wc -l)
        if [ "$OUTPUT_COUNT" -gt 0 ]; then
            ls -lh data/processed/01_etl_output/ 2>/dev/null | grep -E "\.parquet$" | awk '{print "  " $9 " (" $5 ")"}'
        else
            echo "  (No output files yet)"
        fi
    else
        echo "  (Output directory not created yet)"
    fi

    echo ""
    echo "================================================================================"
    echo "  Refreshing in 30 seconds... (Press Ctrl+C to exit)"
    echo "================================================================================"

    sleep 30
done
