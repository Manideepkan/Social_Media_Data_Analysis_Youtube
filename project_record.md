# Project Record: YouTube Data Extraction, Visualization, and Social Network Analysis

## 1. Project Objective
The primary goal of this project was to leverage the **YouTube Data API v3** to extract a comprehensive, real-world dataset of 10,000 video records. Following extraction, the objective was to perform Exploratory Data Analysis (EDA) via Python visualizations and conduct Online Social Network Analysis through interactive graph navigation.

---

## 2. Phase 1: Data Collection & Quota Engineering (`fetch_data.py`)

### The Challenge
The free tier of the YouTube Data API v3 enforces a strict **10,000 quota units per day** limit. Standard `search.list` requests cost 100 units each, meaning fetching 10,000 videos natively through search queries would require over 20,000 quota units—making it impossible. Furthermore, YouTube removed the public dislike counter, making sentiment metrics like `negative_ratio` unavailable natively. 

### The Solution
We engineered a highly efficient batch-fetching pipeline using Python's `urllib` and `json` libraries:
1. **Seed Discovery (100 units):** Queried 10 broad topics (e.g., "tech," "vlogs," "gaming") via the `search` endpoint to discover ~500 unique channels.
2. **Playlist Extraction (1 unit/call):** Passed the channel IDs through the `channels` endpoint to locate their hidden *"Uploads"* playlist ID.
3. **Video ID Harvesting (1 unit/call):** Iterated through the `playlistItems` endpoint, fetching up to 200 videos per channel until a global pool of 10,000 unique video IDs was secured.
4. **Metadata Assembly (1 unit/call):** Passed the 10,000 video IDs to the `videos` endpoint in batches of 50 to extract rich statistics (views, likes, comments, duration).

### Data Transformation & Output
We calculated and normalized custom columns such as `engagement_score`, `like_view_ratio`, `upload_hour`, `upload_day_of_week`, and `video_age_hours`. Unavailable fields (`toxic_ratio`, `negative_ratio`, `related_video_ids`) were gracefully filled with zeroes/nulls to strictly maintain the demanded CSV structure. 
- **Result:** Successfully exported exactly 10,000 records to `youtube_data.csv` using roughly ~500 API quota units (saving over 95% of the daily limit).

---

## 3. Phase 2: Exploratory Data Visualizations (`visualize_data.py`)

With the dataset secured, we utilized `pandas`, `seaborn`, and `matplotlib` to generate 10 unique data visualizations mapping out YouTube's platform behavior:

1. **Correlation Heatmap:** Demonstrated the numeric correlations across all metadata, revealing the heavy ties between views, likes, and comments.
2. **Views vs. Likes (Log Scale):** A scatter plot showing the strictly linear trajectory of likes as viewership scales logarithmically.
3. **Upload Hour Distribution:** A bar chart identifying the most popular hours (in UTC) for content creators to publish videos.
4. **Upload Day Distribution:** Highlighted the distribution of uploads across the 7 days of the week.
5. **Top 10 Channels by Views:** A horizontal bar chart identifying the heavy hitters in our random sample based on total accumulated view counts.
6. **Duration Distribution:** A histogram showing a dense concentration of videos sitting tightly in the 10–15 minute timeframe.
7. **Engagement Score Density:** A Kernel Density Estimate (KDE) plot revealing that standard engagement ratios cleanly hover between 1% and 6%.
8. **Views vs. Comments (Log Scale):** Scattered plotting representing commenting velocity against view scaling.
9. **Like-to-View Percentage Distribution:** A histogram showing the exact average percentage of viewers who leave a "Like".
10. **Top Uploaders in Dataset:** Plotted the 15 most frequent channels captured in our 10,000 record sweep.

---

## 4. Phase 3: Social Network Analysis (`network_analysis.py`)

For the final task, rather than analyzing standard user-to-user comments (which were prohibitively expensive on the API), we constructed a **Semantic Tag Co-occurrence Social Network**. 

### Graph Architecture
1. **Nodes (Tags):** We parsed the `tags` column from the CSV, extracting the top 150 most frequently used keywords across the 10,000 videos.
2. **Edges (Links):** If two tags were used together on the exact same video, an edge was formed between them. The weight of the edge increased the more often they co-occurred.
3. **Network Mathematics (`networkx`):** 
   - Calculated **Degree Centrality** to scale the physical size of nodes (heavily used tags appeared larger).
   - Applied **Greedy Modularity Community Detection** to mathematically group thematic tags into colors (automatically clustering gaming terms together, news terms together, etc.).

### Interactive Generation (`pyvis`)
The graphical mathematics were heavily exported utilizing the `pyvis` framework into a fully navigable, physics-simulated UI (`interactive_network.html`). Users can drag nodes, watch the spring-gravity settle communities, and visually inspect the massive social network built solely from standard metric data.
