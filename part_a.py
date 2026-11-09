"""Part A pipeline. Run from repository root: python part_a.py"""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW, CLEAN = ROOT/'data'/'raw', ROOT/'data'/'clean'
DB, SQL = ROOT/'data'/'instamart.db', ROOT/'sql'/'queries.sql'
OUT = ROOT/'outputs'
for p in (CLEAN, DB.parent, OUT): p.mkdir(parents=True, exist_ok=True)
for name in ('customers.csv','products.csv','orders.csv'):
    if not (RAW/name).exists():
        raise FileNotFoundError(f"Missing data/raw/{name}. Run python generate_data.py first.")

customers_raw = pd.read_csv(RAW/'customers.csv')
products_raw = pd.read_csv(RAW/'products.csv')
orders_raw = pd.read_csv(RAW/'orders.csv')

# Raw audit: dimensions, dtypes, missing counts and percentages.
with (OUT/'part_a_audit.txt').open('w', encoding='utf-8') as f:
    for name, df in [('customers',customers_raw),('products',products_raw),('orders',orders_raw)]:
        f.write(f'\n{name.upper()}\nShape: {df.shape}\nDtypes:\n{df.dtypes}\n')
        audit = pd.DataFrame({'missing_count':df.isna().sum(),
                              'missing_percent':(df.isna().mean()*100).round(2)})
        f.write(audit.to_string()+'\n')
    amount_num = pd.to_numeric(orders_raw.order_amount, errors='coerce')
    delivery_num = pd.to_numeric(orders_raw.delivery_minutes, errors='coerce')
    age_num = pd.to_numeric(customers_raw.age, errors='coerce')
    raw_stats = {
        'raw orders rows':len(orders_raw),
        'exact duplicate rows':int(orders_raw.duplicated().sum()),
        'pending rows':int(orders_raw.status.astype(str).str.lower().eq('pending').sum()),
        'distinct order_id':int(orders_raw.order_id.nunique()),
        'nonnumeric order_amount':int(amount_num.isna().sum()),
        'negative delivery_minutes':int((delivery_num<0).sum()),
        'blank delivery_minutes':int(orders_raw.delivery_minutes.isna().sum()+orders_raw.delivery_minutes.astype(str).str.strip().eq('').sum()),
        'blank customer ages':int(customers_raw.age.isna().sum()+customers_raw.age.astype(str).str.strip().eq('').sum()),
        'ages outside 10-100':int(((age_num<10)|(age_num>100)).sum()),
        'raw city spellings':int(customers_raw.city.nunique()),
        'nonblank raw gender spellings':int(customers_raw.loc[customers_raw.gender.notna() & customers_raw.gender.astype(str).str.strip().ne(''),'gender'].nunique()),
        'blank genders':int(customers_raw.gender.isna().sum()+customers_raw.gender.astype(str).str.strip().eq('').sum())}
    f.write('\nRAW CHECKPOINTS\n')
    for k,v in raw_stats.items(): f.write(f'{k}: {v}\n')

# Clean customers.
customers = customers_raw.copy()
customers['city'] = customers.city.fillna('').astype(str).str.strip().str.title()
g = customers.gender.fillna('').astype(str).str.strip().str.lower()
customers['gender'] = g.map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
customers['age'] = pd.to_numeric(customers.age, errors='coerce')
customers.loc[~customers.age.between(10,100), 'age'] = np.nan
customers['age'] = customers.age.fillna(customers.groupby('city').age.transform('median')).fillna(customers.age.median()).round().astype(int)
customers['signup_date'] = pd.to_datetime(customers.signup_date, errors='coerce').dt.strftime('%Y-%m-%d')

# Clean products.
products = products_raw.copy()
products['price_inr'] = pd.to_numeric(products.price_inr, errors='coerce')
products['product_id'] = products.product_id.astype(str)

# Clean orders in the required sequence.
orders = orders_raw.copy()
orders['order_ts'] = pd.to_datetime(orders.order_ts, errors='coerce')
orders = orders.drop_duplicates().sort_values(['order_id','order_ts'])
orders = orders.drop_duplicates('order_id', keep='last').copy()
orders['quantity'] = pd.to_numeric(orders.quantity, errors='coerce')
orders['discount_pct'] = pd.to_numeric(orders.discount_pct, errors='coerce').fillna(0)
orders['order_amount'] = pd.to_numeric(orders.order_amount, errors='coerce')
orders = orders.merge(products[['product_id','price_inr','category']], on='product_id', how='left', validate='many_to_one')
calculated = (orders.price_inr * orders.quantity * (1-orders.discount_pct/100)).round(2)
orders['order_amount'] = orders.order_amount.fillna(calculated).round(2)
orders['delivery_minutes'] = pd.to_numeric(orders.delivery_minutes, errors='coerce')
orders.loc[orders.delivery_minutes<0, 'delivery_minutes'] = np.nan
completed = orders.status.astype(str).str.lower().eq('completed')
cat_median = orders.loc[completed].groupby('category').delivery_minutes.transform('median')
completed_copy = orders.loc[completed].copy()
completed_copy['cat_median'] = cat_median
orders.loc[completed, 'delivery_minutes'] = orders.loc[completed,'delivery_minutes'].fillna(completed_copy.cat_median.to_numpy())
orders = orders.drop(columns=['price_inr','category'])
orders['order_ts'] = orders.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')

# Write clean CSVs and SQLite database.
customers.to_csv(CLEAN/'customers.csv', index=False)
products.to_csv(CLEAN/'products.csv', index=False)
orders.to_csv(CLEAN/'orders.csv', index=False)
if DB.exists(): DB.unlink()
with sqlite3.connect(DB) as conn:
    customers.to_sql('customers',conn,index=False,if_exists='replace')
    products.to_sql('products',conn,index=False,if_exists='replace')
    orders.to_sql('orders',conn,index=False,if_exists='replace')
    if SQL.exists():
        statements = [s.strip() for s in SQL.read_text(encoding='utf-8').split(';') if s.strip()]
        for i, query in enumerate(statements,1):
            result = pd.read_sql_query(query,conn)
            print(f'\nQ{i}: {len(result)} row(s)')
            print(result.head(10).to_string(index=False))

# Required cleaning checkpoint summary.
cm = orders.status.astype(str).str.lower().eq('completed')
xm = orders.status.astype(str).str.lower().eq('cancelled')
print('\nCLEANING CHECKPOINTS')
print('Clean orders:',len(orders))
print('Completed:',int(cm.sum()),'Cancelled:',int(xm.sum()))
print('Pending:',int(orders.status.astype(str).str.lower().eq('pending').sum()))
print('Missing order_amount:',int(orders.order_amount.isna().sum()))
print('Missing completed delivery_minutes:',int(orders.loc[cm,'delivery_minutes'].isna().sum()))
print('Missing cancelled delivery_minutes:',int(orders.loc[xm,'delivery_minutes'].isna().sum()))
print(f"Completed revenue: INR {orders.loc[cm,'order_amount'].sum():,.2f}")

# pandas Q2 cross-check and city-level named aggregation.
joined = orders.loc[cm].merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id')
pivot = pd.pivot_table(joined,index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
print('\nPANDAS PIVOT TABLE (Q2 CROSS-CHECK)\n',pivot.to_string())
city_summary = joined.groupby('city').agg(total_revenue=('order_amount','sum'),average_order_amount=('order_amount','mean'),order_count=('order_id','count')).round(2)
print('\nCITY SUMMARY\n',city_summary.to_string())

# Distribution, Tukey outliers and correlations.
amounts = orders.loc[cm,'order_amount'].dropna()
q1, med, q3 = amounts.quantile([.25,.5,.75]); iqr=q3-q1
lo, hi = q1-1.5*iqr, q3+1.5*iqr
print('\nORDER AMOUNT EXPLORATION')
print(f'Mean INR {amounts.mean():.2f}; median INR {med:.2f}; skewness {amounts.skew():.2f}')
print(f'Q1 INR {q1:.2f}; Q3 INR {q3:.2f}; lower fence INR {lo:.2f}; upper fence INR {hi:.2f}')
print('Outliers below:',int((amounts<lo).sum()),'above:',int((amounts>hi).sum()))
corr = orders.loc[cm,['order_amount','discount_pct','quantity','delivery_minutes']].apply(pd.to_numeric,errors='coerce').corr()
print('\nCORRELATION MATRIX\n',corr.round(3).to_string())
pairs=[]; cols=list(corr.columns)
for i in range(len(cols)):
    for j in range(i+1,len(cols)): pairs.append((abs(corr.iloc[i,j]),cols[i],cols[j],corr.iloc[i,j]))
if pairs:
    _,a,b,v=max(pairs); print(f'Largest absolute correlation: {a} vs {b}: {v:.3f}')
print('\nPart A complete. Now run: python validate_part_a.py')
