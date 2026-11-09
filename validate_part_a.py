"""Part A reference-check validation. Run: python validate_part_a.py"""
from pathlib import Path
import sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parent
C=ROOT/'data'/'clean'; DB=ROOT/'data'/'instamart.db'
def check(ok,label):
    if not ok: raise AssertionError('FAILED: '+label)
    print('PASS:',label)
def main():
    customers=pd.read_csv(C/'customers.csv'); products=pd.read_csv(C/'products.csv'); orders=pd.read_csv(C/'orders.csv')
    check(len(customers)==1200,'1,200 cleaned customers')
    check(len(products)==40,'40 cleaned products')
    check(len(orders)==12309,'12,309 final orders')
    check(orders.order_id.nunique()==len(orders),'one row per order_id')
    check(not orders.status.astype(str).str.lower().eq('pending').any(),'no pending orders remain')
    check(orders.order_amount.notna().all(),'no missing order_amount values')
    completed=orders.status.astype(str).str.lower().eq('completed')
    cancelled=orders.status.astype(str).str.lower().eq('cancelled')
    check(int(completed.sum())==11110,'11,110 completed orders')
    check(int(cancelled.sum())==1199,'1,199 cancelled orders')
    check(orders.loc[completed,'delivery_minutes'].notna().all(),'completed delivery times are populated')
    check(orders.loc[cancelled,'delivery_minutes'].isna().all(),'cancelled delivery times remain missing')
    revenue=round(pd.to_numeric(orders.loc[completed,'order_amount']).sum(),2)
    check(revenue==2998668.80,'completed revenue is INR 2,998,668.80')
    check(customers.age.between(18,60).all(),'cleaned ages are between 18 and 60')
    check(set(customers.gender.unique())<={'male','female','unknown'},'gender values standardised')
    with sqlite3.connect(DB) as conn:
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        check({'customers','products','orders'}.issubset(tables),'SQLite has all three required tables')
        check(conn.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==12309,'SQLite has 12,309 orders')
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
