import pandas as pd
import numpy as np

np.random.seed(42)

data = []

# Define base year for simulation
base_year = np.random.randint(2010, 2023)

for customer_id in range(1, 3001):
    # Generate fixed base characteristics
    gender = np.random.choice(['M', 'F'])
    initial_age = np.random.randint(18, 60)
    vehicle_initial_age = np.random.randint(0, 5)
    total_customer_time = np.random.randint(5, 11)
    start_year = base_year - total_customer_time
    
    claims_history = 0
    credit_score = np.clip(np.random.normal(700, 50), 300, 850)
    accumulated_violations = 0
    
    # Generate annual history
    for relative_year in range(1, total_customer_time + 1):
        current_year = start_year + relative_year
        
        # Update temporal variables
        age = initial_age + relative_year - 1
        vehicle_age = vehicle_initial_age + relative_year - 1
        
        # Generate annual variables
        annual_mileage = np.random.randint(5000, 35000) + np.random.randint(-1000, 1000)
        annual_violations = np.random.poisson(0.3 + accumulated_violations*0.1)
        accumulated_violations += annual_violations
        
        # Update credit score
        credit_score = np.clip(
            credit_score + np.random.normal(2, 5) - (annual_violations*5) - (claims_history*10),
            300, 850
        )
        
        # Calculate risk
        risk = 0.1 + \
               (vehicle_age * 0.03) + \
               (annual_mileage/50000) + \
               ((800 - credit_score)/500) + \
               (accumulated_violations * 0.2) + \
               (claims_history * 0.4)
        
        claim_probability = 1 / (1 + np.exp(-risk))
        claim_occurred = np.random.binomial(1, claim_probability)
        
        # Add record
        data.append([
            customer_id,
            age,
            gender,
            relative_year,
            current_year,    
            vehicle_age,
            annual_mileage,
            round(credit_score, 2),
            annual_violations,
            claims_history,
            claim_occurred
        ])
        
        claims_history += claim_occurred

columns = [
    'customer_id',
    'age',
    'gender',
    'customer_time_years',
    'year',
    'vehicle_age',
    'annual_mileage',
    'credit_score',
    'traffic_violations_year',
    'claims_history',
    'claim_occurred'
]

df = pd.DataFrame(data, columns=columns)
df = df.sort_values(['customer_id', 'year'])

# Save to Excel
df.to_excel('historical_customer_data.xlsx', index=False)
print("File generated successfully!")