import pandas as pd
import numpy as np
from datetime import datetime, timedelta


print("🚀 Starting Enhanced Data Preparation...")

df = pd.read_csv('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\train.csv', parse_dates=['Date'], low_memory=False)
store = pd.read_csv('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\store.csv')

df = df[df['Store'] == 1].copy()

df.rename(columns={
    'Sales': 'Revenue_EGP',
    'Customers': 'Orders_Delivered',
    'Open': 'Is_Open',
    'Promo': 'Is_White_Friday'
}, inplace=True)

# 3. TIME SHIFT (2015 -> 2026)
max_date_old = df['Date'].max()
target_date = pd.Timestamp('2026-01-14')
time_shift = target_date - max_date_old
df['Date'] = df['Date'] + time_shift


# 2015-2026: ~8x multiplier considering Egyptian pound devaluation
df['Revenue_EGP'] = df['Revenue_EGP'] * 8



df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Egyptian Weekend (Friday=4, Saturday=5)
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [4, 5] else 0)

# Thursday (pre-weekend shopping spike)
df['Is_Thursday'] = (df['DayOfWeek'] == 3).astype(int)

# Ramadan Periods (Accurate dates for 2024-2026)
def get_ramadan_phase(d):
    """
    Returns: 0=Normal, 1=Ramadan_Early, 2=Ramadan_Peak, 3=Pre_Eid
    """
    ramadan_periods = [
        # 2024: Mar 11 - Apr 9
        (pd.Timestamp('2024-03-11'), pd.Timestamp('2024-04-09')),
        # 2025: Mar 1 - Mar 30
        (pd.Timestamp('2025-03-01'), pd.Timestamp('2025-03-30')),
        # 2026: Feb 18 - Mar 19
        (pd.Timestamp('2026-02-18'), pd.Timestamp('2026-03-19'))
    ]
    
    for start, end in ramadan_periods:
        if start.date() <= d.date() <= end.date():
            days_in = (d - start).days
            total_days = (end - start).days
            
            if days_in < 10:
                return 1  # Early Ramadan
            elif days_in < total_days - 5:
                return 2  # Peak Ramadan
            else:
                return 3  # Pre-Eid (last 5 days - shopping surge)
    return 0

df['Ramadan_Phase'] = df['Date'].apply(get_ramadan_phase)
df['Is_Ramadan'] = (df['Ramadan_Phase'] > 0).astype(int)

# Eid al-Fitr (3 days after Ramadan ends)
def get_eid_fitr(d):
    eid_periods = [
        (pd.Timestamp('2024-04-10'), pd.Timestamp('2024-04-12')),
        (pd.Timestamp('2025-03-31'), pd.Timestamp('2025-04-02')),
        (pd.Timestamp('2026-03-20'), pd.Timestamp('2026-03-22'))
    ]
    for start, end in eid_periods:
        if start.date() <= d.date() <= end.date():
            return 1
    return 0

df['Is_Eid_Fitr'] = df['Date'].apply(get_eid_fitr)

# Eid al-Adha (approximate - 70 days after Eid al-Fitr)
def get_eid_adha(d):
    eid_periods = [
        (pd.Timestamp('2024-06-15'), pd.Timestamp('2024-06-18')),
        (pd.Timestamp('2025-06-06'), pd.Timestamp('2025-06-09')),
        (pd.Timestamp('2026-05-27'), pd.Timestamp('2026-05-30'))
    ]
    for start, end in eid_periods:
        if start.date() <= d.date() <= end.date():
            return 1
    return 0

df['Is_Eid_Adha'] = df['Date'].apply(get_eid_adha)

# Coptic Christmas (Jan 7)
df['Is_Coptic_Christmas'] = ((df['Month'] == 1) & (df['Day'] == 7)).astype(int)

# Payday Effect (Stronger impact)
df['Is_Payday_Early'] = df['Day'].apply(lambda d: 1 if 1 <= d <= 5 else 0)
df['Is_Payday_Late'] = df['Day'].apply(lambda d: 1 if 25 <= d <= 31 else 0)
df['Is_Payday'] = (df['Is_Payday_Early'] | df['Is_Payday_Late']).astype(int)



# Summer (June-August) - heat affects fresh produce
df['Is_Summer'] = df['Month'].apply(lambda m: 1 if m in [6, 7, 8] else 0)

# Winter (December-February) - increased dairy/hot beverages
df['Is_Winter'] = df['Month'].apply(lambda m: 1 if m in [12, 1, 2] else 0)

# School Year (Sep-May, excluding Ramadan)
df['Is_School_Season'] = df['Month'].apply(lambda m: 1 if 9 <= m <= 5 else 0)


products = {
    "Bakery": {
        "revenue_share": 0.08,
        "price_per_unit": 35,
        "ramadan_multiplier": 0.8,
        "summer_multiplier": 0.95,
        "weekend_boost": 1.15
    },
    "Dairy": {
        "revenue_share": 0.15,
        "price_per_unit": 60,
        "ramadan_multiplier": 2.5,
        "summer_multiplier": 0.85,
        "weekend_boost": 1.10
    },
    "Beverages": {
        "revenue_share": 0.10,
        "price_per_unit": 25,
        "ramadan_multiplier": 1.8,
        "summer_multiplier": 1.4,
        "weekend_boost": 1.20
    },
    "Fresh_Produce": {
        "revenue_share": 0.12,
        "price_per_unit": 20,
        "ramadan_multiplier": 1.5,
        "summer_multiplier": 0.90,
        "weekend_boost": 1.25
    },
    "Frozen_Foods": {
        "revenue_share": 0.09,
        "price_per_unit": 45,
        "ramadan_multiplier": 1.3,
        "summer_multiplier": 1.0,
        "weekend_boost": 1.05
    },
    "Snacks": {
        "revenue_share": 0.11,
        "price_per_unit": 15,
        "ramadan_multiplier": 2.0,
        "summer_multiplier": 1.1,
        "weekend_boost": 1.30
    },
    "Household": {
        "revenue_share": 0.14,
        "price_per_unit": 50,
        "ramadan_multiplier": 1.2,
        "summer_multiplier": 1.0,
        "weekend_boost": 1.00
    },
    "Personal_Care": {
        "revenue_share": 0.10,
        "price_per_unit": 70,
        "ramadan_multiplier": 0.9,
        "summer_multiplier": 1.05,
        "weekend_boost": 1.10
    }
}

product_data = []

for idx, row in df.iterrows():
    base_revenue = row['Revenue_EGP']
    
    for product_name, profile in products.items():
        # Base allocation
        product_revenue = base_revenue * profile['revenue_share']
        
        # Apply multipliers
        if row['Is_Ramadan']:
            product_revenue *= profile['ramadan_multiplier']
        
        if row['Is_Summer']:
            product_revenue *= profile['summer_multiplier']
        
        if row['Is_Weekend']:
            product_revenue *= profile['weekend_boost']
        
        # Add realistic noise (±15%)
        noise = np.random.uniform(0.85, 1.15)
        product_revenue *= noise
        
        # Calculate units
        units = int(product_revenue / profile['price_per_unit'])
        
        product_data.append({
            'Date': row['Date'],
            'Product_Category': product_name,
            'Revenue_EGP': product_revenue,
            'Units_Sold': units,
            'Price_Per_Unit': profile['price_per_unit'],
            **{col: row[col] for col in [
                'Is_Open', 'Is_White_Friday', 'Is_Weekend', 'Is_Thursday',
                'Ramadan_Phase', 'Is_Ramadan', 'Is_Eid_Fitr', 'Is_Eid_Adha',
                'Is_Coptic_Christmas', 'Is_Payday', 'Is_Payday_Early', 
                'Is_Payday_Late', 'Is_Summer', 'Is_Winter', 'Is_School_Season',
                'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear'
            ]}
        })

df_products = pd.DataFrame(product_data)


df = df[df['Is_Open'] == 1].copy()
df_products = df_products[df_products['Is_Open'] == 1].copy()

q_high = df['Revenue_EGP'].quantile(0.995)
df = df[df['Revenue_EGP'] <= q_high].copy()



store_output = 'C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\egyptian_sales_store_level.csv'
df.to_csv(store_output, index=False)
print(f"✅ Store-level data saved to {store_output}")
print(f"   Shape: {df.shape}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

product_output = 'C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\egyptian_sales_product_level.csv'
df_products.to_csv(product_output, index=False)
print(f"✅ Product-level data saved to {product_output}")
print(f"   Shape: {df_products.shape}")
print(f"   Products: {df_products['Product_Category'].nunique()}")



print("\n📊 DATA SUMMARY")
print("="*60)
print(f"Total Days: {len(df)}")
print(f"Avg Daily Revenue: {df['Revenue_EGP'].mean():,.0f} EGP")
print(f"Revenue Range: {df['Revenue_EGP'].min():,.0f} - {df['Revenue_EGP'].max():,.0f} EGP")
print(f"\nRamadan Days: {df['Is_Ramadan'].sum()}")
print(f"Weekend Days: {df['Is_Weekend'].sum()}")
print(f"Payday Periods: {df['Is_Payday'].sum()}")
print(f"White Friday Events: {df['Is_White_Friday'].sum()}")

print("\n📦 PRODUCT BREAKDOWN (Avg Daily Units)")
print("="*60)
product_summary = df_products.groupby('Product_Category').agg({
    'Units_Sold': 'mean',
    'Revenue_EGP': 'mean'
}).round(0)
print(product_summary)

print("\n🌙 RAMADAN IMPACT (Revenue Multiplier)")
print("="*60)
for product in df_products['Product_Category'].unique():
    normal = df_products[(df_products['Product_Category'] == product) & 
                         (df_products['Is_Ramadan'] == 0)]['Revenue_EGP'].mean()
    ramadan = df_products[(df_products['Product_Category'] == product) & 
                          (df_products['Is_Ramadan'] == 1)]['Revenue_EGP'].mean()
    if normal > 0:
        multiplier = ramadan / normal
        print(f"{product:20s}: {multiplier:.2f}x")

