#!/bin/bash
# Initialize error flag
HAS_ERROR=0
ERROR_MESSAGE=""

# Initialize success message
SUCCESS_MESSAGE="Job Summary:\n"

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
  echo -e "Subject:ERROR - Cannot activate Python venv\n\nFailed to activate the Python virtual environment." | mstmp andrei.aldescu@yahoo.com
  exit 1
fi

# Run dacia.py
echo "Running dacia.py..."
python dacia.py
if [ $? -eq 0 ]; then
  echo "dacia.py completed successfully."
  SUCCESS_MESSAGE+="- dacia.py: SUCCESS\n"
else
  echo "Error: dacia.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- dacia.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- dacia.py: FAILED\n"
fi
echo "-----------------------------------"

# Clear the content in resume.txt
echo "Clearing resume.txt..."
> resume.txt
if [ $? -eq 0 ]; then
  echo "resume.txt cleared successfully."
  SUCCESS_MESSAGE+="- resume.txt: CLEARED\n"
else
  echo "Error: Could not clear resume.txt."
  HAS_ERROR=1
  ERROR_MESSAGE+="- Failed to clear resume.txt.\n"
  SUCCESS_MESSAGE+="- resume.txt: CLEAR FAILED\n"
fi
echo "-----------------------------------"

# Run get_ads.py
echo "Running get_ads.py..."
python get_ads.py
if [ $? -eq 0 ]; then
  echo "get_ads.py completed successfully."
  SUCCESS_MESSAGE+="- get_ads.py: SUCCESS\n"
else
  echo "Error: get_ads.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- get_ads.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- get_ads.py: FAILED\n"
fi
echo "-----------------------------------"

# Run transform.py
echo "Running transform.py..."
python transform.py
if [ $? -eq 0 ]; then
  echo "transform.py completed successfully."
  SUCCESS_MESSAGE+="- transform.py: SUCCESS\n"
else
  echo "Error: transform.py encountered an issue."
  HAS_ERROR=1
  ERROR_MESSAGE+="- transform.py failed to execute properly.\n"
  SUCCESS_MESSAGE+="- transform.py: FAILED\n"
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
    SUCCESS_MESSAGE+="Git: Changes committed successfully.\n"
  fi

  # Push the changes to the main branch
  git push origin main
  if [ $? -ne 0 ]; then
    HAS_ERROR=1
    ERROR_MESSAGE+="Failed to push changes to repository.\n"
  else
    SUCCESS_MESSAGE+="Git: Changes pushed to repository.\n"
  fi
fi

echo "Git operations completed."
echo "-----------------------------------"

# Deactivate the virtual environment
deactivate

# Send email based on status
if [ $HAS_ERROR -eq 1 ]; then
  SUBJECT="ERROR - Script processing pipeline"
  BODY="Errors occurred during the processing pipeline:\n\n$ERROR_MESSAGE\n\nFull Job Summary:\n$SUCCESS_MESSAGE"
else
  SUBJECT="SUCCESS - Script processing pipeline complete"
  BODY="The processing pipeline completed successfully:\n\n$SUCCESS_MESSAGE"
fi

# Send the email
echo -e "Subject:$SUBJECT\n\n$BODY" | msmtp andrei.aldescu@yahoo.com

exit $HAS_ERROR
