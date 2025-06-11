# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:58:12 2025

@author: Abiodun Tiamiyu

About Bibitor
Bibitor, LLC ia a retail wine and spirits company, with 79 or 80 locations within
the fictional state of Lincoln. Sales ranges from 420-450 million dollars and cost
of goods sold can range from 300-350 million dollars. The total number of records 
in the datasets is 15 million. This includes sales transaction (12.5-13 million
records), purchase transactions (2.3-2.5 million records) and inventory.

Executive Question and Requests:
1. Which Vendor has bthe largest freight cost?
2. Inventory tracking or auditing:
    a.  Which inventory items in EndInvFINAL12312016.csv were not present in 
        BegInvFINAL12312016.
    b.  Which items existed both at the beginning and end of the period.
    c.  How many inventory items were carried over from beginning to end of the 
        inventory.
    d.  What percentage of inventory at the end was already preswent at the 
        beginning.

Business Task:
    Bibitor has asked the team to complete due dilligence on their wine and spirits 
business looking at the data for their beginning and ending inventory.

Datasets Given:
    - BegInvFINAL12312016.csv
    - EndInvFINAL12312016.csv
    - 2017PurchasePriceDec.csv
    - InvoicePurchases12312016.csv
"""
import pandas as pd
    # Import CSV files (datasets from BIbitor) and assigning a new name to them
BegInvDec = pd.read_csv("BegInvFINAL12312016.csv")
EndInvDec = p = pd.read_csv("SalesFINAL12312016.csv")
VendorInvoiceDec = pd.read_csv("InvoicePurchases12312016.csv")

""" Question 1: Which Vendor has the largest freight cost?

"""
#%%
    # Listig out the variables in VendorInvoiceDec file
VendorInvoiceDec.info()
print(VendorInvoiceDec.info())
    # Filter: Dollars > 100 and Quantity <= 1000
df = VendorInvoiceDec
filtered_df = df[(df["Dollars"] > 100) & (df["Quantity"] <= 1000)]

    # Group by VendorNumber
grouped_df = filtered_df.groupby("VendorNumber", as_index = False)["Freight"].sum()

    # Rename the Freight Column to TotalFreight
grouped_df.rename(columns = {"Freight": "TotalFreight"}, inplace = True)

FreightSummary = grouped_df.sort_values(by = "TotalFreight", ascending = False)
FreightSummary.to_csv("FreightSummary.csv", index = False)                                                    

"""
Result: 
Based on our result, we found that Vendor #2561 has incurred the largest freight
cost which is $3,176.81
"""
#%%
"""
Question 2: Inventory Tracking/ Auditing
    # 2a. Which Inventory items in the ending inventory file were not present in 
beginning inventory file

"""
    # Find new inventory added
beg_df = BegInvDec
end_df = EndInvDec
#%%
new_inventory_df = pd.merge(end_df, beg_df, on = "InventoryId", how = "left", 
                            indicator = True)

new_items_df = new_inventory_df[new_inventory_df["_merge"] == "left_only"]
print("Newly added inventory items:")
print(new_items_df[["InventoryId"]])
new_items_df.to_csv("new_items_df.csv", index = False)

"""
Result:
    We found that 49,513 inventory items were not present the beginning inventory
    
"""

#%%
"""
Question 2b: Which items existed both at the beginning and end of the period?
"""
    # Items that appers in both
common_items_df = pd.merge(end_df, beg_df, on = "InventoryId", how = "inner")
print("Inventory items present in both periods:")
print(common_items_df[["InventoryId"]])
common_items_df.to_csv("common_items_df.csv", index = False)

"""
Result:
174,976 inventory items existed both at the beginning and end of the period
"""
#%%
"""
Question 2c: How many inventory items were carried over from beginning to end?

"""
carryover_count = len(common_items_df)
print(carryover_count)
"""
Result:
    174,976 inventory items were carried over from beginning to the end
"""

#%%
"""
Question 2d: What percentage of inventory at the end that was already present at the
beginning?
"""
total_end_count = len(end_df)
percent_retained = (carryover_count/total_end_count)*100
print(percent_retained)
"""
Result:
    The percentage of inventory at the end that was already present at the 
    beginning is ~ 77.94%
"""
