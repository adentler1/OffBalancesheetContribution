# -*- coding: utf-8 -*-

"""

Created on Wed May 28 11:29:03 2025

 

@author: up19653

 

 

https://www.api-football.com/sports

https://dashboard.api-football.com/

https://www.api-football.com/documentation-v3

 

"""


#%%
# 📋 Configuration Variables
API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"     # Your API-Football key
# API_KEY ="f80f0d51fae7375ecc9cfef166e5fcf7"#bundesbank account

# League & Season
LEAGUE_NAME = "Bundesliga"
COUNTRY_NAME = "Germany"
SEASON = 2023

API_URL = "https://v3.football.api-sports.io"    # Base API URL 
# Rate limiting & batching
TIME_BETWEEN_REQUESTS = 10          # Seconds to wait between API calls (rate-limit buffer)
# File paths
SAVE_COUNTRY_DIR         = f"{COUNTRY_NAME}_{LEAGUE_NAME}"
SAVE_INFO_DIR         = f"{COUNTRY_NAME}/{SEASON}"
LEAGUE_FILE      = "league.json"
TEAMS_FILE       = "teams.json"
TEAMS_STATUS     = "teams_status.json"
PLAYERS_FILE     = "players.json"
PLAYERS_STATUS   = "players_status.json"
FIXTURES_FILE    = "fixtures.json"
EVENTS_FILE      = "match_events.json"
STATUS_FILE      = "fixtures_status.json"

#%%
# 🛠 Imports & Setup
import os
import time
import json
import requests

# Ensure save directory exists
os.makedirs(SAVE_COUNTRY_DIR, exist_ok=True)
os.makedirs(SAVE_INFO_DIR, exist_ok=True)

# Prepare request headers
headers = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}
headersS = {
    'x-apisports-key': API_KEY,
}

#%%
def get_requests_left():
    response = requests.get(API_URL + '/status', headers=headers)
    if response.status_code == 200:
        out = response.json()['response']['requests']['limit_day']-response.json()['response']['requests']['current']
        return out
    else:
        raise Exception(f'API Error: {response.status_code} - {response.text}')        
requests_remaining = get_requests_left()

# 🔧 Helper Functions
def api_get(endpoint, params=None):
    """
    Perform GET with global rate-limit tracking.b
    Updates requests_remaining from response headers.
    """
    global requests_remaining
    url = f"{API_URL}{endpoint}"
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    # Update remaining requests
    requests_remaining = int(resp.headers.get(
        'x-ratelimit-requests-remaining', requests_remaining
    ))
    return resp.json()


def load_json(filename):
    """Load JSON from SAVE_INFO_DIR if exists, else return empty dict."""
    path = os.path.join(SAVE_INFO_DIR, filename)
    return json.load(open(path)) if os.path.exists(path) else {}


def save_json(filename, data):
    """Save JSON to SAVE_INFO_DIR."""
    path = os.path.join(SAVE_INFO_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

#%%
# 1️⃣ Fetch & Save League ID
path = os.path.join(SAVE_COUNTRY_DIR, LEAGUE_FILE)
if os.path.exists(path):
    league_data = json.load(open(path))
else:
    league_data = api_get('/leagues', {'name': LEAGUE_NAME, 'country': COUNTRY_NAME})
    with open(path, 'w') as f:
        json.dump(league_data, f, indent=2)

league_id = league_data['response'][0]['league']['id']
print(f"League ID: {league_id} | Requests left: {requests_remaining}")

#%%
# 2️⃣ Fetch & Save Teams Info
teams_data = load_json(TEAMS_FILE)
teams_status = load_json(TEAMS_STATUS)

if not teams_data:
    teams_data = api_get('/teams', {'league': league_id, 'season': SEASON})
    if teams_data['results']==0:
        print('HANDLE ERROR BY NOT CONDUCTING ANY OF THE FOLLOWING REQUESTS. STATE THAT DATA IS NOT YET AVAILABLE')
    # initialize status flags
    teams_status = {t['team']['name']: False for t in teams_data['response']}

for t in teams_data['response']:
    name = t['team']['name']
    if teams_status.get(name):
        continue  # already fetched
    if requests_remaining <= 0:
        print("Daily request limit reached during teams download.")
        break
    # Save team metadata
    teams_status[name] = True
    print(f"Saved metadata for team {name} | Requests left: {requests_remaining}")
    time.sleep(TIME_BETWEEN_REQUESTS)
# Save after loop
save_json(TEAMS_FILE, teams_data)
save_json(TEAMS_STATUS, teams_status)

#%%
# 3️⃣ Fetch & Save Player Metadata
players_data = load_json(PLAYERS_FILE)
players_status = load_json(PLAYERS_STATUS)

# Initialize status if first run
if not players_status and teams_data:
    players_status = {t['team']['name']: False for t in teams_data['response']}

for t in teams_data['response']:
    name = t['team']['name']
    team_id = t['team']['id']
    if players_status.get(name):
        continue
    if requests_remaining <= 0:
        print("Daily request limit reached during players download.")
        break
    # Fetch players for this team & season
    resp = api_get('/players', {'team': team_id, 'season': SEASON})
    players_list = []
    # API paginates players; accumulate if > 1 page
    page, total_pages = 1, resp['paging']['total']
    while page <= total_pages:
        data = api_get('/players', {'team': team_id, 'season': SEASON, 'page': page})
        for p in data['response']:
            players_list.append({
                'id': p['player']['id'],
                'name': p['player']['name'],
                'age': p['player'].get('age'),
                'position': p['statistics'][0]['games'].get('position')
            })
        page += 1
        time.sleep(TIME_BETWEEN_REQUESTS)
    players_data[name] = players_list
    players_status[name] = True
    print(f"Fetched {len(players_list)} players for {name} | Requests left: {requests_remaining}")

# Save after all teams
save_json(PLAYERS_FILE, players_data)
save_json(PLAYERS_STATUS, players_status)

#%%
# 4️⃣ Fetch & Save Fixture IDs
fixtures_data = load_json(FIXTURES_FILE)
if not fixtures_data:
    fixtures_data = api_get('/fixtures', {'league': league_id, 'season': SEASON})
    save_json(FIXTURES_FILE, fixtures_data)

fixture_ids = [f['fixture']['id'] for f in fixtures_data['response']]
print(f"Total fixtures: {len(fixture_ids)} | Requests left: {requests_remaining}")

#%%
# 5️⃣ Fetch Fixture Events Individually (free tier)
match_events   = load_json(EVENTS_FILE)
fixtures_status = load_json(STATUS_FILE)

fixtures_remaining = len(fixture_ids) - sum(
    1 for fid in fixture_ids
    if fixtures_status.get(str(fid)) and len(match_events[str(fid)]['lineups']) == 2 and len(match_events[str(fid)]['events']) > 0
)

for fid in fixture_ids:
    key = str(fid)
    # Skip already-downloaded fixtures
    if fixtures_status.get(key) and len(match_events[key]['lineups']) == 2 and len(match_events[key]['events']) > 0:
        continue

    # Check rate limit
    if requests_remaining <= 0:
        print("Daily request limit reached during fixtures download.")
        break

    # Display progress
    print(f"Fixtures remaining: {fixtures_remaining}")

    # Fetch lineups for this fixture
    lineup_resp = api_get('/fixtures/lineups', {'fixture': fid})
    time.sleep(TIME_BETWEEN_REQUESTS)

    # Fetch events (goals, subs) for this fixture
    events_resp = api_get('/fixtures/events', {'fixture': fid})
    time.sleep(TIME_BETWEEN_REQUESTS)

    # Store responses
    match_events[key] = {
        'lineups': lineup_resp.get('response', []),
        'events': events_resp.get('response', [])
    }
    
    if len(match_events[key]['lineups']) == 2 and len(match_events[key]['events']) > 0:
        fixtures_status[key] = True
    else:
        fixtures_status[key] = False
        

    fixtures_remaining -= 1
    print(f"Saved fixture {key} | Requests left: {requests_remaining}")

    # Persist progress after each fixture
    save_json(EVENTS_FILE, match_events)
    save_json(STATUS_FILE, fixtures_status)

print("All available downloads complete for today.")