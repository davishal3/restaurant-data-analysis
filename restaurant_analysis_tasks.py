import os
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN


# ============================================================
# Basic setup
# ============================================================

FILE_PATH = "/mnt/data/Dataset .csv"   # Change this if your CSV is stored somewhere else
OUTPUT_DIR = "/mnt/data/restaurant_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ============================================================
# Load dataset
# ============================================================

# Reading the dataset
# Keeping a copy so the original data stays unchanged

df = pd.read_csv(FILE_PATH)
data = df.copy()

# Removing extra spaces from column names just to be safe
# This avoids small naming issues later in the code

data.columns = data.columns.str.strip()

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print("\nColumns:")
print(list(data.columns))
print("\nMissing values:")
print(data.isnull().sum())
print("\n")


# ============================================================
# Helper columns for cuisine analysis
# ============================================================

# Splitting cuisine strings into clean lists
# Example: 'North Indian, Chinese' -> ['North Indian', 'Chinese']

data["Cuisines"] = data["Cuisines"].fillna("Unknown")
data["Cuisine List"] = data["Cuisines"].apply(
    lambda x: [c.strip() for c in str(x).split(",") if c.strip()]
)

# Creating a flat list of all cuisines
all_cuisines = data["Cuisine List"].explode()


# ============================================================
# Task 1: Top Cuisines
# ============================================================

print("=" * 80)
print("TASK 1: TOP CUISINES")
print("=" * 80)

# Counting the most common individual cuisines
# Since one restaurant can serve more than one cuisine,
# each cuisine is counted separately here

top_cuisines = all_cuisines.value_counts().head(3)
print("Top 3 most common cuisines:")
print(top_cuisines)
print()

# Calculating what percentage of restaurants serve each of the top cuisines
# A restaurant is counted once for a cuisine if that cuisine appears in its list

total_restaurants = len(data)
for cuisine, count in top_cuisines.items():
    percentage = (data["Cuisine List"].apply(lambda x: cuisine in x).sum() / total_restaurants) * 100
    print(f"{cuisine}: {percentage:.2f}% of restaurants")
print("\n")


# ============================================================
# Task 2: City Analysis
# ============================================================

print("=" * 80)
print("TASK 2: CITY ANALYSIS")
print("=" * 80)

# Finding the city with the highest number of restaurants
city_counts = data["City"].value_counts()
city_highest_restaurants = city_counts.idxmax()
print(f"City with the highest number of restaurants: {city_highest_restaurants}")
print(f"Number of restaurants: {city_counts.max()}")
print()

# Average rating by city
city_avg_rating = data.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False)
print("Average rating for restaurants in each city:")
print(city_avg_rating)
print()

# City with the highest average rating
city_highest_avg_rating = city_avg_rating.idxmax()
print(f"City with the highest average rating: {city_highest_avg_rating}")
print(f"Average rating: {city_avg_rating.max():.2f}")
print("\n")


# ============================================================
# Task 3: Price Range Distribution
# ============================================================

print("=" * 80)
print("TASK 3: PRICE RANGE DISTRIBUTION")
print("=" * 80)

# Counting restaurants in each price range category
price_counts = data["Price range"].value_counts().sort_index()
price_percentages = (price_counts / total_restaurants) * 100

print("Restaurant count by price range:")
print(price_counts)
print()
print("Percentage by price range:")
print(price_percentages.round(2))
print()

# Creating a bar chart for price range distribution
plt.figure(figsize=(8, 5))
plt.bar(price_counts.index.astype(str), price_counts.values)
plt.title("Distribution of Price Ranges")
plt.xlabel("Price Range")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task_3_price_range_distribution.png"), dpi=300)
plt.close()

print(f"Chart saved: {os.path.join(OUTPUT_DIR, 'task_3_price_range_distribution.png')}")
print("\n")


# ============================================================
# Task 4: Online Delivery
# ============================================================

print("=" * 80)
print("TASK 4: ONLINE DELIVERY")
print("=" * 80)

# Percentage of restaurants offering online delivery
online_delivery_counts = data["Has Online delivery"].value_counts()
online_delivery_percentage = (online_delivery_counts / total_restaurants) * 100

print("Online delivery percentage:")
print(online_delivery_percentage.round(2))
print()

# Comparing average ratings for restaurants with and without online delivery
avg_rating_online_delivery = data.groupby("Has Online delivery")["Aggregate rating"].mean()
print("Average ratings by online delivery status:")
print(avg_rating_online_delivery.round(2))
print("\n")


# ============================================================
# Task 5: Restaurant Ratings
# ============================================================

print("=" * 80)
print("TASK 5: RESTAURANT RATINGS")
print("=" * 80)

# Plotting the distribution of aggregate ratings
plt.figure(figsize=(9, 5))
plt.hist(data["Aggregate rating"], bins=10, edgecolor="black")
plt.title("Distribution of Aggregate Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task_5_rating_distribution.png"), dpi=300)
plt.close()

print(f"Chart saved: {os.path.join(OUTPUT_DIR, 'task_5_rating_distribution.png')}")

# Finding the most common rating range using bins
rating_bins = [0, 1, 2, 3, 4, 5]
rating_labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
data["Rating Range"] = pd.cut(
    data["Aggregate rating"],
    bins=rating_bins,
    labels=rating_labels,
    include_lowest=True,
    right=True,
)

rating_range_counts = data["Rating Range"].value_counts().sort_index()
most_common_rating_range = rating_range_counts.idxmax()

print("Rating range counts:")
print(rating_range_counts)
print()
print(f"Most common rating range: {most_common_rating_range}")

# Average votes
average_votes = data["Votes"].mean()
print(f"Average number of votes received by restaurants: {average_votes:.2f}")
print("\n")


# ============================================================
# Task 6: Cuisine Combination
# ============================================================

print("=" * 80)
print("TASK 6: CUISINE COMBINATION")
print("=" * 80)

# Standardizing cuisine combinations so that the same pair is counted together
# Example: 'Chinese, North Indian' and 'North Indian, Chinese' should be treated the same

def normalize_cuisine_combination(cuisine_list):
    cleaned = sorted(set([c.strip() for c in cuisine_list if c.strip()]))
    return ", ".join(cleaned)


data["Cuisine Combination"] = data["Cuisine List"].apply(normalize_cuisine_combination)
combination_counts = data["Cuisine Combination"].value_counts()

print("Most common cuisine combinations:")
print(combination_counts.head(10))
print()

# Checking which cuisine combinations tend to have higher ratings
combination_rating = (
    data.groupby("Cuisine Combination")
    .agg(
        Restaurant_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Votes=("Votes", "mean"),
    )
    .sort_values(by=["Average_Rating", "Restaurant_Count"], ascending=[False, False])
)

# Filtering combinations with at least 5 restaurants for a fairer comparison
combination_rating_filtered = combination_rating[combination_rating["Restaurant_Count"] >= 5]

print("Cuisine combinations with higher ratings (minimum 5 restaurants):")
print(combination_rating_filtered.head(10))
print("\n")


# ============================================================
# Task 7: Geographic Analysis
# ============================================================

print("=" * 80)
print("TASK 7: GEOGRAPHIC ANALYSIS")
print("=" * 80)

# Keeping only rows with valid latitude and longitude
geo_data = data.dropna(subset=["Latitude", "Longitude"]).copy()

print(f"Restaurants with valid coordinates: {len(geo_data)}")
print()

# Creating an interactive map
# Marker clustering helps when many restaurants are close together
if not geo_data.empty:
    map_center = [geo_data["Latitude"].mean(), geo_data["Longitude"].mean()]
    restaurant_map = folium.Map(location=map_center, zoom_start=5)
    marker_cluster = MarkerCluster().add_to(restaurant_map)

    # Adding markers for restaurants
    # Using a limited number for better performance in the HTML map
    for _, row in geo_data.head(2000).iterrows():
        popup_text = (
            f"<b>{row['Restaurant Name']}</b><br>"
            f"City: {row['City']}<br>"
            f"Cuisines: {row['Cuisines']}<br>"
            f"Rating: {row['Aggregate rating']}"
        )
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=popup_text,
        ).add_to(marker_cluster)

    map_path = os.path.join(OUTPUT_DIR, "task_7_restaurant_map.html")
    restaurant_map.save(map_path)
    print(f"Interactive map saved: {map_path}")
    print()

    # Basic cluster detection using DBSCAN
    # Coordinates are converted to radians for distance-based grouping
    coords = np.radians(geo_data[["Latitude", "Longitude"]].to_numpy())
    kms_per_radian = 6371.0088
    epsilon_km = 5
    epsilon = epsilon_km / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=10, algorithm="ball_tree", metric="haversine")
    geo_data["Cluster"] = db.fit_predict(coords)

    cluster_summary = (
        geo_data[geo_data["Cluster"] != -1]
        .groupby("Cluster")
        .agg(
            Restaurant_Count=("Restaurant ID", "count"),
            Main_City=("City", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            Main_Locality=("Locality", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        )
        .sort_values(by="Restaurant_Count", ascending=False)
    )

    if not cluster_summary.empty:
        print("Largest restaurant clusters found in the dataset:")
        print(cluster_summary.head(10))
    else:
        print("No strong geographic clusters were found with the current settings.")
else:
    print("No valid latitude and longitude values found for map analysis.")
print("\n")


# ============================================================
# Task 8: Restaurant Chains
# ============================================================

print("=" * 80)
print("TASK 8: RESTAURANT CHAINS")
print("=" * 80)

# Treating repeated restaurant names as possible chains
# This is a simple approach and works well for this dataset
chain_counts = data["Restaurant Name"].value_counts()
chains = chain_counts[chain_counts > 1]

print("Possible restaurant chains present in the dataset:")
print(chains.head(20))
print()

# Analyzing ratings and popularity of chains
chain_analysis = (
    data[data["Restaurant Name"].isin(chains.index)]
    .groupby("Restaurant Name")
    .agg(
        Outlet_Count=("Restaurant ID", "count"),
        Average_Rating=("Aggregate rating", "mean"),
        Average_Votes=("Votes", "mean"),
        Total_Votes=("Votes", "sum"),
        Cities_Present=("City", "nunique"),
    )
    .sort_values(by=["Outlet_Count", "Average_Rating", "Total_Votes"], ascending=[False, False, False])
)

print("Chain analysis:")
print(chain_analysis.head(20))
print("\n")


# ============================================================
# Saving main result tables as CSV files
# ============================================================

# Saving key outputs so they can be used directly in reports or dashboards

top_cuisines.to_csv(os.path.join(OUTPUT_DIR, "task_1_top_cuisines.csv"), header=["Count"])
city_avg_rating.to_csv(os.path.join(OUTPUT_DIR, "task_2_city_average_rating.csv"), header=["Average Rating"])
price_percentages.to_csv(os.path.join(OUTPUT_DIR, "task_3_price_range_percentages.csv"), header=["Percentage"])
combination_rating_filtered.to_csv(os.path.join(OUTPUT_DIR, "task_6_cuisine_combination_ratings.csv"))
chain_analysis.to_csv(os.path.join(OUTPUT_DIR, "task_8_restaurant_chain_analysis.csv"))

print("All tasks completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")
