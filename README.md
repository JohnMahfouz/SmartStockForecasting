# 🛒 SmartStock AI  
## AI-Powered Inventory Forecasting for the Egyptian Retail Market

SmartStock AI is an intelligent demand-forecasting system that helps grocery stores and supermarkets **order the right quantity of products every day** — eliminating guesswork, reducing waste, and maximizing revenue.

It combines **machine learning**, **Egyptian calendar intelligence (Ramadan, Eid, weekends)**, and **product shelf-life awareness** to deliver **accurate, risk-adjusted inventory recommendations**.

---

## 🎯 Problem Statement

Retail inventory decisions are often based on intuition or simple averages. This leads to:

- ❌ Over-ordering → expired products → wasted money  
- ❌ Under-ordering → stock-outs → lost sales  
- ❌ Ignoring seasonal and cultural demand spikes  

In **Egyptian markets**, demand fluctuates heavily due to:
- Ramadan & Eid
- Payday shopping
- Friday/Saturday weekends

Traditional methods fail to capture these patterns.

---

## ✅ Solution: SmartStock AI

SmartStock AI replaces guessing with **data-driven decisions**.

It predicts:
- 📈 **Total store revenue**
- 📦 **Product-level demand (units & revenue)**
- ⚖️ **Recommended order quantity adjusted for expiry risk**

The system is designed to be:
- Accurate  
- Practical  
- Easy to use (no ML background required)

---

## 🧠 How It Works (Simple Explanation)

### 1️⃣ Learn From the Past 📚
The AI studies historical sales and learns patterns like:
- Fridays have higher demand
- Ramadan increases food sales
- Payday causes spending spikes
- Seasonal product behavior

---

### 2️⃣ Predict the Future 🔮
You provide:
- Date
- Ramadan / Eid status
- Recent sales
- Product category

The AI predicts:
- Tomorrow’s total revenue
- Product-specific demand
- Required stock quantity

---

### 3️⃣ Adjust for Risk ⚠️
Predictions are adjusted using **product shelf life**:

| Product Type | Shelf Life | Ordering Strategy |
|-------------|-----------|------------------|
| Fresh Bread | 1 day | Order ~85% (very cautious) |
| Dairy (Yogurt) | 12 days | Order close to prediction |
| Canned Food | 6+ months | Order ~110% (safe to stock) |


## 📈 Real-World Example

### **Input**
- **Date:** Friday during Ramadan  
- **Yesterday’s Sales:** 45,000 EGP  
- **Product:** Dairy (Yogurt)

### **AI Output**
- **Store Forecast:** 52,000 EGP  
- **Yogurt Demand:** 325 units  
- **Recommended Order:** 300 units  
- **Risk Level:** Medium (12-day shelf life)

### **Result**
✔ Minimal waste  
✔ No stock-out  
✔ Maximum efficiency  

---

## 🔍 What Makes It Smart?

### 🇪🇬 Egyptian Market Intelligence
- Friday & Saturday weekends (not Sunday)
- Ramadan & Eid demand surges
- End-of-month payday shopping behavior

---

### 📦 Product Intelligence
- Shelf-life-aware ordering decisions
- Category-specific demand modeling

---

### 📊 Pattern Recognition
The AI automatically learns patterns such as:
- Weekend stock-up behavior
- Ramadan night consumption spikes
- Seasonal beverage demand increases

---

## 🧪 Technology Stack

- **Python**
- **XGBoost** — high-accuracy regression engine
- **Pandas / NumPy** — data processing & analysis
- **Scikit-learn** — preprocessing & evaluation
- **Streamlit** — interactive web dashboard

---

## 🎯 Who Is This For?

### 🏪 Small Stores
- Reduce expired inventory
- Avoid stock-outs
- Make confident daily ordering decisions

---

### 🏬 Chain Stores
- Scales across multiple branches
- Centralized monitoring & analytics
- Consistent forecasting logic
