# FreshMart Demand Forecasting System Data Dictionary

This preliminary data dictionary documents the four source files used for the FreshMart demand forecasting project. Field names and data types reflect the supplied datasets.

## 1. annex1.csv - Item and Category Master

| Field | Data Type | Description | Project Use |
|---|---|---|---|
| `Item Code` | Integer | Unique code identifying the vegetable item | Links item information to transaction, price, and loss-rate data |
| `Item Name` | Text | Name of the vegetable item | Item identification and reporting |
| `Category Code` | Integer | Code identifying the vegetable category | Category-level analysis |
| `Category Name` | Text | Name of the vegetable category | Category-level analysis and reporting |

**Records:** 251  
**Fields:** 4

## 2. annex2.csv - Transaction Data

| Field | Data Type | Description | Project Use |
|---|---|---|---|
| `Date` | Date | Date of the transaction | Demand trends and seasonality |
| `Time` | Time | Time of the transaction | Transaction timing and demand patterns |
| `Item Code` | Integer | Code identifying the vegetable sold or returned | Links transactions to item information |
| `Quantity Sold (kilo)` | Decimal | Quantity recorded for the transaction in kilograms | Primary demand measure |
| `Unit Selling Price (RMB/kg)` | Decimal | Selling price per kilogram | Price-demand analysis |
| `Sale or Return` | Text | Indicates whether the transaction was a sale or return | Data preparation and demand calculation |
| `Discount (Yes/No)` | Text | Indicates whether a discount applied to the transaction | Potential explanatory variable |

**Records:** 878,503  
**Fields:** 7  
**Date range:** July 1, 2020 to June 30, 2023

## 3. annex3.csv - Wholesale Price Data

| Field | Data Type | Description | Project Use |
|---|---|---|---|
| `Date` | Date | Date of the wholesale price observation | Aligns wholesale prices with demand |
| `Item Code` | Integer | Code identifying the vegetable item | Links prices to item information |
| `Wholesale Price (RMB/kg)` | Decimal | Wholesale price per kilogram | Price-related forecasting and procurement analysis |

**Records:** 55,982  
**Fields:** 3  
**Date range:** July 1, 2020 to June 30, 2023

## 4. annex4.csv - Loss Rate Data

| Field | Data Type | Description | Project Use |
|---|---|---|---|
| `Item Code` | Integer | Code identifying the vegetable item | Links loss rates to item information |
| `Item Name` | Text | Name of the vegetable item | Item identification |
| `Loss Rate (%)` | Decimal | Percentage loss associated with the item | Procurement-range calculations and spoilage analysis |

**Records:** 251  
**Fields:** 3

## Data Relationships

`Item Code` provides the main link between the item master, transaction, wholesale price, and loss-rate data. The transaction, wholesale price, and loss-rate data can therefore be combined with the item and category information for demand analysis and forecasting.
