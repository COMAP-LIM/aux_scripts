#!/bin/bash

LOG_DIR=/mn/stornext/d22/cmbco/comap/protodir/logs
SCRIPT_DIR=/mn/stornext/d22/cmbco/comap/protodir/comap_aux/cron_jobs

# Cron doesn't support dynamic variable assigment from stuff like the "date" command, so we need this wrapper script to ensure every script is passed the same datetime.
YESTERDAY_MONTH=$(date -d "yesterday" +%Y-%m)
YESTERDAY_DAY=$(date -d "yesterday" +%Y-%m-%d)

# 1. Transfering yesterdays level1 files from OVRO to a temp directory.
# 2. Compress the level1 files in another temp directory, then move them to the default level1 directory.
# 3. Run the weather-NN on new l1 files.
# 4. Run scan-detect to create new runlists containing the new l1 files.
# 5. Run the tsys database, to make calibration files for the new l1 files.
# 6. Run l2gen on all available files, to get l2 files from the new data.
$SCRIPT_DIR/transfer_level1_for_day.sh $YESTERDAY_MONTH $YESTERDAY_DAY >> $LOG_DIR/cronlog_daily_transfers.log 2> >(tee -a $LOG_DIR/cronlog_daily_transfers.log >&2);\
$SCRIPT_DIR/daily_compression_parallel.sh >> $LOG_DIR/cronlog_daily_compression.log 2> >(tee -a $LOG_DIR/cronlog_daily_compression.log >&2);\
$SCRIPT_DIR/weather_nn.sh >> $LOG_DIR/cronlog_weather.log 2> >(tee -a $LOG_DIR/cronlog_weather.log >&2);\
$SCRIPT_DIR/daily_scandetect.sh >> $LOG_DIR/cronlog_scandetect.log 2> >(tee -a $LOG_DIR/cronlog_scandetect.log >&2);\
$SCRIPT_DIR/daily_tsys_database.sh $YESTERDAY_MONTH >> $LOG_DIR/cronlog_tsys_database.log 2> >(tee -a $LOG_DIR/cronlog_tsys_database.log >&2);\
# $SCRIPT_DIR/daily_l2gen.sh >> $LOG_DIR/cronlog_l2gen.log 2> >(tee -a $LOG_DIR/cronlog_l2gen.log >&2);\