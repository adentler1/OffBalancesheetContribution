#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:15:09 2025

@author: alexander
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:00:41 2025

@author: alexander
"""

# -*- coding: utf-8 -*-
import os
import time
import json
import requests

class FootballDataFetcher:
    API_URL = "https://v3.football.api-sports.io"
    TIME_BETWEEN_REQUESTS = 10

    def __init__(self, API_KEY, LEAGUE_NAME, COUNTRY_NAME, SEASON):
        self.API_KEY = API_KEY
        self.LEAGUE_NAME = LEAGUE_NAME
        self.COUNTRY_NAME = COUNTRY_NAME
        self.SEASON = SEASON
        self.SAVE_DIR = f"{COUNTRY_NAME}_{LEAGUE_NAME}_{SEASON}"
        self.headers = {'x-rapidapi-key': self.API_KEY, 'x-rapidapi-host': 'v3.football.api-sports.io'}
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.requests_remaining = self.get_requests_left()

    def get_requests_left(self):
        response = requests.get(f"{self.API_URL}/status", headers=self.headers)
        response.raise_for_status()
        data = response.json()['response']['requests']
        return data['limit_day'] - data['current']

    def api_get(self, endpoint, params=None):
        url = f"{self.API_URL}{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        self.requests_remaining = int(resp.headers.get('x-ratelimit-requests-remaining', self.requests_remaining))
        return resp.json()

    def load_json(self, filename):
        path = os.path.join(self.SAVE_DIR, filename)
        return json.load(open(path)) if os.path.exists(path) else {}

    def save_json(self, filename, data):
        path = os.path.join(self.SAVE_DIR, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def fetch_league_data(self):
        league_data = self.load_json("league.json")
        if not league_data:
            league_data = self.api_get('/leagues', {'name': self.LEAGUE_NAME, 'country': self.COUNTRY_NAME})
            self.save_json("league.json", league_data)
        league_id = league_data['response'][0]['league']['id']
        return league_id

    def fetch_teams_data(self, league_id):
        teams_data = self.load_json("teams.json")
        teams_status = self.load_json("teams_status.json")
        if not teams_data:
            teams_data = self.api_get('/teams', {'league': league_id, 'season': self.SEASON})
            if teams_data['results'] == 0:
                print('HANDLE ERROR BY NOT CONDUCTING ANY OF THE FOLLOWING REQUESTS. STATE THAT DATA IS NOT YET AVAILABLE')
                return None
            teams_status = {t['team']['name']: False for t in teams_data['response']}

        for t in teams_data['response']:
            name = t['team']['name']
            if teams_status.get(name):
                continue
            if self.requests_remaining <= 0:
                print("Daily request limit reached during teams download.")
                break
            teams_status[name] = True
            time.sleep(self.TIME_BETWEEN_REQUESTS)

        self.save_json("teams.json", teams_data)
        self.save_json("teams_status.json", teams_status)
        return teams_data

    def run_fetch(self):
        league_id = self.fetch_league_data()
        teams_data = self.fetch_teams_data(league_id)
        if teams_data is None:
            return  # No further action needed due to unavailable data

        # Additional methods like players, fixtures, etc. would follow similarly.

# Example usage
if __name__ == '__main__':
    API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"
    fetcher = FootballDataFetcher(API_KEY, "Bundesliga", "Germany", 2023)
    fetcher.run_fetch()
