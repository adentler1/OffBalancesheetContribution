# Soccer Analytics Dashboard

Interactive dashboard for analyzing player contributions using econometric methods.

## Quick Start

### 1. Install Python (if needed)

You need Python 3.9 or higher. Check with:
```bash
python3 --version
```

### 2. Create a virtual environment (recommended)

```bash
# Create environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

### 5. Open in browser

The dashboard will open automatically, or go to:
```
http://localhost:8501
```

---

## What's Inside

| File | Description |
|------|-------------|
| `dashboard.py` | Main Streamlit application |
| `analysis_results.db` | Player contribution data (2010-2023) |
| `player_profiles.db` | Player metadata (age, nationality, transfers) |
| `requirements.txt` | Python dependencies |

## Data Coverage

- **78,080** player-season records
- **31** leagues across **25** countries
- Seasons: **2010-2023**

## Dashboard Pages

1. **Home** - Methodology explanation and key insights
2. **Contribution Analysis** - Explore player contributions by position/team
3. **Player Career** - Track individual player trajectories
4. **Distribution Comparison** - Compare leagues
5. **Transfer Networks** - Visualize talent flow
6. **Persistence** - Test predictive power of contributions
7. **Free Agents** - Find disappeared talent

## The Metric

**Lasso Contribution** = estimated change in team goal difference (per 90 minutes) when a player is on the pitch, controlling for all teammates and opponents.

- Positive = team performs better with this player
- Zero = average impact
- Negative = team performs worse

---

## Troubleshooting

**"Database not found"**
Make sure `analysis_results.db` is in the same folder as `dashboard.py`.

**Port already in use**
Run on a different port:
```bash
streamlit run dashboard.py --server.port 8502
```

**Slow first load**
The first load caches data. Subsequent interactions are faster.
