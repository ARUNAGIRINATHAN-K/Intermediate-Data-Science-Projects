# 🩸 Blood Transfusion Dataset 

This project uses the **Blood Transfusion Service Center Dataset** to practice **data analysis and visualization**.  
It helps us understand the donor behavior and learn how to use Python for analytics.  

---

## 📂 Dataset Information

The dataset contains information about blood donors.  
Each row is one donor, with these columns:

1. **Recency** - Months since the last donation  
2. **Frequency** - Total number of donations  
3. **Monetary** - Total blood donated (in c.c.)  
4. **Time** - Months since the first donation  
5. **Class(Target)** - Whether the donor gave blood in March 2007  
   - `1` = Donated  
   - `2` = Did not donate  

---

## ⚙️ Steps in the Notebook

### 1. Import Libraries
We use:
- **pandas** → handle tabular data (like Excel in Python)  
- **matplotlib & seaborn** → create charts and plots  

### 2. Load the Dataset
```python
df = pd.read_csv("Blood Transfusion Service data.csv")
```
This loads the CSV file into a DataFrame (like a table).

### 3. Basic Data Info

- `df.head()` → shows the first few rows
- `df.info()` → data types (numbers, categories, etc.)
- `df.describe()` → summary statistics (mean, std, min, max)

### 4. Target Variable (Class)

We check how many donors vs non-donors are there.  
This tells us if the dataset is balanced or not.  

We also make a **pie chart** to see donation percentages.

### 5. Feature Distributions

We plot histograms for:

- **Recency** → Do most donors donate recently or long ago?  
- **Frequency** → Are donors frequent or one-time?  
- **Monetary** → How much blood do people usually donate?  
- **Time** → How long donors stay with the service.

### 6. Compare Donors vs Non-Donors

Using **boxplots**, we compare:

- **Recency** (how recent donations are) between donors and non-donors.  

This helps us see patterns:  
👉 People who donated in March 2007 usually have lower `Recency` (recent donors).

### 7. Correlation Heatmap

We check if features are related:

- `Frequency` and `Monetary` are usually highly correlated (more donations = more blood donated).  

This helps us understand relationships between variables.

