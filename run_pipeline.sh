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

# Pull the latest changes from the repository
echo "Pulling latest changes from git repository..."
git pull origin main
if [ $? -ne 0 ]; then
  HAS_ERROR=1
  ERROR_MESSAGE+="Failed to pull latest changes from git repository.\n"
  echo -e "Subject:ERROR - Cannot pull from git\n\nFailed to pull the latest changes from the git repository." | msmtp andrei.aldescu@yahoo.com
  echo "Continuing with local version..."
fi

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

# Clear the content in resume.txt
echo "Clearing resume.json..."
> resume.json
if [ $? -eq 0 ]; then
  echo "resume.json cleared successfully."
  SUCCESS_MESSAGE+="- resume.json: CLEARED\n"
else
  echo "Error: Could not clear resume.txt."
  HAS_ERROR=1
  ERROR_MESSAGE+="- Failed to clear resume.txt.\n"
  SUCCESS_MESSAGE+="- resume.txt: CLEAR FAILED\n"
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

# Git operations
echo "Starting Git operations..."
# Add all changed files
git add .

# Check if there are changes to commit
if git diff-index --quiet HEAD --; then
  echo "No changes to commit"
  SUCCESS_MESSAGE+="Git: No changes detected to commit.\n"
else
  # Commit the changes with a message
  git commit -m "Updated data $(date +"%Y-%m-%d %H:%M:%S")"
  if [ $? -ne 0 ]; then
    HAS_ERROR=1
    ERROR_MESSAGE+="Failed to commit changes.\n"
  else
    SUCCESS_MESSAGE+="Git: Changes committed locally (no push).\n"
  fi
fi

echo "Git operations completed."
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
