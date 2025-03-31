import csv
import random
import datetime
import math

def generate_mock_data(output_file="car_data.csv"):
    # Define date range
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2025, 3, 24)  # Today's date
    
    # Define car categories and their base popularity (higher number = more popular)
    car_categories = {
        "Dokker": 400,
        "Duster": 700,  # Most popular
        "Jogger": 500,
        "Lodgy": 300,   # Least popular
        "Logan": 550
    }
    
    # Calculate number of days
    delta = end_date - start_date
    num_days = delta.days + 1
    
    # Create data rows
    data_rows = []
    
    # Seasonal factors (Q1, Q2, Q3, Q4)
    seasonal_factors = [0.85, 1.1, 0.9, 1.15]  # Lower in winter, higher in summer & holidays
    
    # Weekly pattern (Monday to Sunday)
    weekly_patterns = [0.9, 0.85, 0.9, 1.0, 1.2, 1.3, 0.75]  # Weekend spike, low on Sunday
    
    # Growth trends for each model (annual percentage growth)
    annual_growth = {
        "Dokker": -0.05,  # Declining slightly
        "Duster": 0.15,   # Strong growth
        "Jogger": 0.25,   # New model with rapid growth
        "Lodgy": -0.15,   # Being phased out
        "Logan": 0.05     # Stable with slight growth
    }
    
    # Generate data for each day and each car category
    for day in range(num_days):
        current_date = start_date + datetime.timedelta(days=day)
        formatted_date = current_date.strftime("%m/%d/%Y")
        
        # Calculate year progress for trend calculation
        years_passed = day / 365.0
        
        # Get day of week (0 = Monday, 6 = Sunday)
        day_of_week = current_date.weekday()
        
        # Get quarter (0-3)
        quarter = (current_date.month - 1) // 3
        
        # Special events - holidays with increased activity
        special_event = 1.0
        if (current_date.month == 12 and current_date.day >= 15) or \
           (current_date.month == 1 and current_date.day <= 10):
            special_event = 1.3  # Holiday season boost
        elif (current_date.month == 7 and current_date.day >= 1 and current_date.day <= 15):
            special_event = 1.25  # Summer vacation boost
        
        for category, base_count in car_categories.items():
            # Apply growth trend
            trend_factor = 1 + (annual_growth[category] * years_passed)
            
            # Calculate base count with trend
            trended_base = base_count * trend_factor
            
            # Apply seasonal and weekly patterns
            seasonal_factor = seasonal_factors[quarter]
            weekly_factor = weekly_patterns[day_of_week]
            
            # Add some randomness (±10%)
            random_factor = random.uniform(0.9, 1.1)
            
            # Calculate final count
            count = int(trended_base * seasonal_factor * weekly_factor * special_event * random_factor)
            
            # Ensure count is within 1-999 range
            count = max(1, min(count, 999))
            
            data_rows.append([formatted_date, category, count])
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(["Date", "Category", "Count"])
        
        # Write data rows
        csv_writer.writerows(data_rows)
    
    print(f"Generated {len(data_rows)} records in {output_file}")
    print(f"- Date range: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}")
    print(f"- Categories: {', '.join(car_categories.keys())}")
    print(f"- Each model has the following trend:")
    for car, growth in annual_growth.items():
        print(f"  * {car}: {growth*100:+.1f}% annual change")

if __name__ == "__main__":
    generate_mock_data()