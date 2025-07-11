#!/bin/bash
# Initialize error flag
HAS_ERROR=0
ERROR_MESSAGE=""

# Initialize success message
SUCCESS_MESSAGE="Job Summary:\n"

# Record start time
START_TIME=$(date +%s)

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the  directory for better context.
cd "$SCRIPT_DIR" || exit 1



# Activate the virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"
if [ $? -ne 0 ]; then
  HAS_ERROR=1
  ERROR_MESSAGE+="Failed to activate virtual environment.\n"
  echo -e "Subject:ERROR - Cannot activate Python venv\n\nFailed to activate the Python virtual environment." | msmtp andrei.aldescu@yahoo.com
  exit 1
fi

# Run extract_dacia.py
echo "Running extract_dacia.py..."
python extract_dacia.py
if [ $? -eq 0 ]; then
  echo "extract_dacia.py completed successfully."
  SUCCESS_MESSAGE+="- extract_dacia.py: SUCCESS\n"
else
  echo "Error: extract_dacia.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- dacia.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- dacia.py: FAILED\n"
fi
echo "-----------------------------------"

# Clear the content in resume.json
echo "Clearing resume.json..."
> resume.json
if [ $? -eq 0 ]; then
  echo "resume.json cleared successfully."
  SUCCESS_MESSAGE+="- resume.json: CLEARED\n"
else
  echo "Error: Could not clear resume.json."
  HAS_ERROR=1
  ERROR_MESSAGE+="- Failed to clear resume.json.\n"
  SUCCESS_MESSAGE+="- resume.json: CLEAR FAILED\n"
fi
echo "-----------------------------------"

# Run extract_dacia_ads.py
echo "Running extract_dacia_ads.py..."
python extract_dacia_ads.py
if [ $? -eq 0 ]; then
  echo "extract_dacia_ads.py completed successfully."
  SUCCESS_MESSAGE+="- extract_dacia_ads.py: SUCCESS\n"
else
  echo "Error: extract_dacia_ads.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- extract_dacia_ads.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- extract_dacia_ads.py: FAILED\n"
fi
echo "-----------------------------------"

# Run transform_load.py
echo "Running transform_load.py..."
python transform_load.py
if [ $? -eq 0 ]; then
  echo "transform_load.py completed successfully."
  SUCCESS_MESSAGE+="- transform_load.py: SUCCESS\n"
else
  echo "Error: transform_load.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- transform_load.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- transform_load.py: FAILED\n"
fi
echo "-----------------------------------"

# Clean resume_detailed.json
echo "Clearing resume_detailed.json..."
> resume_detailed.json
if [ $? -eq 0 ]; then
  echo "resume_detailed.json cleared successfully."
  SUCCESS_MESSAGE+="- resume_detailed.json: CLEARED\n"
else
  echo "Error: Could not clear resume_detailed.json."
  HAS_ERROR=1
  ERROR_MESSAGE+="- Failed to clear resume_detailed.json.\n"
  SUCCESS_MESSAGE+="- resume_detailed.json: CLEAR FAILED\n"
fi
echo "-----------------------------------"

# Run the generate_detail_ad_urls.py script.
# arguments: 
# all: all ads that have a href
# today: ads posted today
# default: today

echo "Running generate_detail_ad_urls.py..."
python generate_detail_ad_urls.py
if [ $? -eq 0 ]; then
  echo "generate_detail_ad_urls.py completed successfully."
  SUCCESS_MESSAGE+="- generate_detail_ad_urls.py: SUCCESS\n"
else
  echo "Error: generate_detail_ad_urls.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- generate_detail_ad_urls.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- generate_detail_ad_urls.py: FAILED\n"
fi
echo "-----------------------------------"

# Run the extract_dacia_detailed_ad.py script
python extract_dacia_detailed_ad.py
if [ $? -eq 0 ]; then
  echo "extract_dacia_detailed_ad.py completed successfully."
  SUCCESS_MESSAGE+="- extract_dacia_detailed_ad.py: SUCCESS\n"
else
  echo "Error: extract_dacia_detailed_ad.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- extract_dacia_detailed_ad.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- extract_dacia_detailed_ad.py: FAILED\n"
fi
echo "-----------------------------------"  

# Run the transform_load.py script
python transform_load.py
if [ $? -eq 0 ]; then
  echo "transform_load.py completed successfully."
  SUCCESS_MESSAGE+="- transform_load.py: SUCCESS\n"
else
  echo "Error: transform_load.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- transform_load.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- transform_load.py: FAILED\n"
fi
echo "-----------------------------------"


# Deactivate the virtual environment
deactivate

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
HOURS=$((EXECUTION_TIME / 3600))
MINUTES=$(( (EXECUTION_TIME % 3600) / 60 ))
SECONDS=$((EXECUTION_TIME % 60))
TIME_MESSAGE="Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "$TIME_MESSAGE"

# Send email based on status
if [ $HAS_ERROR -eq 1 ]; then
  SUBJECT="ERROR - Script processing pipeline"
  BODY="Errors occurred during the processing pipeline:\n\n$ERROR_MESSAGE\n\nFull Job Summary:\n$SUCCESS_MESSAGE\n\n$TIME_MESSAGE"
else
  SUBJECT="SUCCESS - Script processing pipeline complete"
  BODY="The processing pipeline completed successfully:\n\n$SUCCESS_MESSAGE\n\n$TIME_MESSAGE"
fi

# Send the email
echo -e "Subject:$SUBJECT\n\n$BODY" | msmtp andrei.aldescu@yahoo.com

exit $HAS_ERROR
