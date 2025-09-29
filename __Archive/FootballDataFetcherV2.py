#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:00:41 2025

@author: alexander
"""

import os
import time
import json
import requests
import pandas as pd

class FootballDataFetcher:
    def __init__(self, api_key, league_name, country_name, season):
        self.api_key = api_key
        self.league_name = league_name
        self.country_name = country_name
        self.season = season
        self.api_url = "https://v3.football.api-sports.io"
        self.rate_limit = 10
        self.requests_remaining = None

        self.save_country_dir = f"{self.country_name}_{self.league_name}"
        self.save_info_dir = f"{self.country_name}_{self.league_name}/{self.season}"

        os.makedirs(self.save_country_dir, exist_ok=True)
        os.makedirs(self.save_info_dir, exist_ok=True)

        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }

        self.files = {
            "league": "league.json",
            "teams": "teams.json",
            "teams_status": "teams_status.json",
            "players": "players.json",
            "players_status": "players_status.json",
            "fixtures": "fixtures.json",
            "match_events": "match_events.json",
            "fixtures_status": "fixtures_status.json"
        }

    def get_requests_left(self):
        """Fetch the number of API requests remaining for today."""
        response = requests.get(f'{self.api_url}/status', headers=self.headers)
        response.raise_for_status()
        data = response.json()
        if data.get('errors') and any(data['errors'].values()):
            raise RuntimeError(f"API error: {data['errors']}")
        # Calculate remaining requests for the day
        requests_left = data['response']['requests']['limit_day'] - data['response']['requests']['current']
        return requests_left

    def api_get(self, endpoint, params=None):
        """Wrapper for GET requests to the football API with quota checking."""
        # **Graceful exit check**: If no requests remaining, do not attempt further API calls.
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            print("Daily request limit reached. Cannot fetch more data from the API.")
            # Return an empty result to signal no data fetched.
            return {"results": 0, "response": []}

        url = f"{self.api_url}{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()

        # Update the remaining requests count from response headers (if provided)
        remaining_header = resp.headers.get('x-ratelimit-requests-remaining')
        if remaining_header is not None:
            try:
                self.requests_remaining = int(remaining_header)
            except (ValueError, TypeError):
                # If the header is missing or not an integer, keep the current requests_remaining.
                pass

        return resp.json()

    def load_json(self, filename):
        """Load JSON data from a file if it exists, otherwise return an empty dict."""
        path = os.path.join(self.save_info_dir, filename)
        return json.load(open(path)) if os.path.exists(path) else {}

    def save_json(self, filename, data):
        """Save JSON data to a file (pretty-printed)."""
        path = os.path.join(self.save_info_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def fetch_league_data(self):
        """Fetch basic league information and cache it if not already cached."""
        path = os.path.join(self.save_country_dir, self.files["league"])

        if os.path.exists(path):
            # Load cached data if available
            with open(path, "r", encoding="utf-8") as f:
                league_data = json.load(f)
        else:
            # Call the API for league information, then cache the result
            league_data = self.api_get("/leagues", {"name": self.league_name, "country": self.country_name})
            # **Check for empty response** (e.g., invalid league or API limit reached)
            if league_data.get('results') == 0 or not league_data.get('response'):
                print(f"No league data found for '{self.league_name}' in {self.country_name}.")
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(league_data, f, ensure_ascii=False, indent=2)

        # Ensure we have a valid response entry to return
        if not league_data.get('response'):
            print(f"No league data available for '{self.league_name}' in {self.country_name}.")
            return None
        return league_data['response'][0]

    def fetch_league_id(self):
        """Retrieve the league ID for the specified league and country."""
        data = self.fetch_league_data()
        if data is None:
            return None
        return data['league']['id']

    def fetch_teams(self, league_id):
        """Fetch team data for the league season, using cache if available."""
        teams_data = self.load_json(self.files["teams"])
        teams_status = self.load_json(self.files["teams_status"])

        if not teams_data:
            teams_data = self.api_get('/teams', {'league': league_id, 'season': self.season})
            # **Check for empty team data** (e.g., data not available or token limit reached)
            if teams_data.get('results') == 0 or not teams_data.get('response'):
                print("No team data available (possibly due to API limit or no data for this season).")
                print("Skipping remaining data fetch for this league/season.")
                return None, None
            # Initialize team processing status for each team
            teams_status = {t['team']['name']: False for t in teams_data['response']}

        # Process each team (if not already processed)
        for t in teams_data.get('response', []):
            name = t['team']['name']
            if teams_status.get(name):
                continue  # Skip teams already marked as processed
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                # **Graceful exit** if request limit reached during team processing
                print("Daily request limit reached during team download. Stopping team data fetch.")
                break

            teams_status[name] = True  # Mark team as processed
            print(f"Processed team {name} | Requests left: {self.requests_remaining}")
            time.sleep(self.rate_limit)  # Rate limiting delay

        # Save teams data and status to disk (caching progress)
        self.save_json(self.files["teams"], teams_data)
        self.save_json(self.files["teams_status"], teams_status)
        return teams_data, teams_status

    def fetch_players(self, teams_data, teams_status):
        """Fetch players data for each team in the league season."""
        players_data = self.load_json(self.files["players"])
        players_status = self.load_json(self.files["players_status"])

        if not players_status and teams_data:
            # Initialize all teams as not processed in players_status
            players_status = {t['team']['name']: False for t in teams_data['response']}

        # Iterate through each team to fetch its players
        for t in teams_data['response']:
            name = t['team']['name']
            team_id = t['team']['id']
            if players_status.get(name):
                continue  # Skip if this team's players have been fetched
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                print("Daily request limit reached during players download. Stopping player data fetch.")
                break

            # Fetch first page of players for the team
            resp = self.api_get('/players', {'team': team_id, 'season': self.season})
            if resp.get('results') == 0 or not resp.get('response'):
                # **Empty players response** (possibly no data or token limit)
                print(f"No player data returned for team {name}. Possibly out of requests.")
                print("Skipping remaining player downloads for this league/season.")
                break

            players_list = []
            page = 1
            total_pages = resp.get('paging', {}).get('total', 1)

            # Add players from the first page
            for p in resp.get('response', []):
                players_list.append({
                    'id': p['player']['id'],
                    'name': p['player']['name'],
                    'age': p['player'].get('age'),
                    'position': p['statistics'][0]['games'].get('position') if p.get('statistics') else None
                })

            page += 1
            # If more pages of players data exist, fetch them
            while page <= total_pages:
                if self.requests_remaining is not None and self.requests_remaining <= 0:
                    print("Daily request limit reached during multi-page player download. Stopping further player data fetch.")
                    break
                data = self.api_get('/players', {'team': team_id, 'season': self.season, 'page': page})
                if data.get('results') == 0 or not data.get('response'):
                    # **No more data or token limit reached mid-pagination**
                    print(f"Empty player data response on page {page} for team {name}. Stopping further pages for this team.")
                    break

                for p in data.get('response', []):
                    players_list.append({
                        'id': p['player']['id'],
                        'name': p['player']['name'],
                        'age': p['player'].get('age'),
                        'position': p['statistics'][0]['games'].get('position') if p.get('statistics') else None
                    })
                page += 1
                time.sleep(self.rate_limit)

                # Continue fetching until all pages are processed or stopped by break
            # End of pagination loop

            players_data[name] = players_list
            players_status[name] = True
            print(f"Fetched {len(players_list)} players for {name} | Requests left: {self.requests_remaining}")

            # If we broke out of the loop early due to request limit, break out of team loop as well
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                break

        # Save the players data and status
        self.save_json(self.files["players"], players_data)
        self.save_json(self.files["players_status"], players_status)

    def fetch_fixtures(self, league_id):
        """Fetch all fixture IDs for the league season."""
        fixtures_data = self.load_json(self.files["fixtures"])
        if not fixtures_data:
            fixtures_data = self.api_get('/fixtures', {'league': league_id, 'season': self.season})
            # **Check for empty fixtures data**
            if fixtures_data.get('results') == 0 or not fixtures_data.get('response'):
                print("No fixtures data available (possibly due to API limit or no data for this season).")
                print("Skipping fixture events fetch for this league/season.")
                # Cache an empty result to avoid repeated API calls
                self.save_json(self.files["fixtures"], fixtures_data)
                return []
            # Save fixtures list to cache
            self.save_json(self.files["fixtures"], fixtures_data)
        # Extract fixture IDs if available
        if not fixtures_data.get('response'):
            return []
        return [f['fixture']['id'] for f in fixtures_data['response']]

    def fetch_fixture_events(self, fixture_ids):
        """Fetch lineups and events for each fixture in the list."""
        match_events = self.load_json(self.files["match_events"])
        fixtures_status = self.load_json(self.files["fixtures_status"])
        total = len(fixture_ids)

        for fid in fixture_ids:
            key = str(fid)
            # Skip if this fixture's data is already fetched and complete
            if key in match_events and match_events[key].get('lineups') and match_events[key].get('events'):
                continue
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                print("Daily request limit reached during fixture events download. Stopping further fixture events fetch.")
                break

            # Fetch lineup for the fixture
            lineup = self.api_get('/fixtures/lineups', {'fixture': fid})
            if lineup.get('results') == 0 or not lineup.get('response'):
                print(f"No lineup data for fixture {fid} (possibly due to API limit or no data).")
                print("Skipping remaining fixture events for this league/season.")
                break
            time.sleep(self.rate_limit)

            # Fetch events for the fixture
            events = self.api_get('/fixtures/events', {'fixture': fid})
            if events.get('results') == 0 or not events.get('response'):
                print(f"No events data for fixture {fid} (possibly due to API limit or no data).")
                print("Skipping remaining fixture events for this league/season.")
                break
            time.sleep(self.rate_limit)

            # Store the fetched lineup and events
            match_events[key] = {
                'lineups': lineup.get('response', []),
                'events': events.get('response', [])
            }
            fixtures_status[key] = True

            done = sum(1 for f in fixture_ids if fixtures_status.get(str(f)))
            remaining = total - done
            print(f"Saved fixture {key} | Requests left: {self.requests_remaining} | {remaining} fixtures remaining")

            # Save progress after each fixture
            self.save_json(self.files["match_events"], match_events)
            self.save_json(self.files["fixtures_status"], fixtures_status)

            # If the request limit is reached mid-way, break out
            if self.requests_remaining is not None and self.requests_remaining <= 0:
                break

    def run_fetch(self):
        """Run the data fetching pipeline for the configured league and season."""
        # Initialize the remaining requests count for the day
        self.requests_remaining = self.get_requests_left()
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            print("No API requests remaining for today. Exiting data fetch for this league/season.")
            return

        league_id = self.fetch_league_id()
        if league_id is None:
            # League not found or no data available, stop further processing
            print(f"League '{self.league_name}' in {self.country_name} not found or data unavailable.")
            return

        teams_data, teams_status = self.fetch_teams(league_id)
        if teams_data is None:
            # No team data (empty response or limit reached), stop further processing
            return

        self.fetch_players(teams_data, teams_status)
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            # Stop if we ran out of requests during player fetch
            print("Request limit reached before fetching fixtures. Stopping further data fetch for this league/season.")
            return

        fixture_ids = self.fetch_fixtures(league_id)
        if not fixture_ids:
            # No fixtures data, stop before fetching events
            return

        self.fetch_fixture_events(fixture_ids)
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            # If we ran out of requests during fixture events
            print("Request limit reached during fixture events fetch. Data fetch for this league/season ended early.")
            return

        print(f"All available downloads for the {self.season} season of the {self.league_name} in {self.country_name} are complete.")

    def league_info(self):
        """Compile league info and coverage data into a pandas DataFrame."""
        data = self.fetch_league_data()
        if data is None:
            # Return empty DataFrame if league data couldn’t be retrieved
            return pd.DataFrame()

        league_info = {
            'league_id': data['league']['id'],
            'league_name': data['league']['name'],
            'country': data['country']['name'],
            'country_code': data['country']['code']
        }

        # Flatten season coverage details into a list of dicts
        seasons_data = []
        for season in data['seasons']:
            season_flat = {
                'year': season['year'],
                'start_date': season['start'],
                'end_date': season['end'],
                'current': season['current'],
                'fixtures_events': season['coverage']['fixtures']['events'],
                'fixtures_lineups': season['coverage']['fixtures']['lineups'],
                'fixtures_statistics': season['coverage']['fixtures']['statistics_fixtures'],
                'players_statistics': season['coverage']['fixtures']['statistics_players'],
                'standings': season['coverage']['standings'],
                'players': season['coverage']['players'],
                'top_scorers': season['coverage']['top_scorers'],
                'top_assists': season['coverage']['top_assists'],
                'top_cards': season['coverage']['top_cards'],
                'injuries': season['coverage']['injuries'],
                'predictions': season['coverage']['predictions'],
                'odds': season['coverage']['odds'],
            }
            season_flat.update(league_info)
            seasons_data.append(season_flat)

        # Create DataFrame from the compiled season data
        df_seasons = pd.DataFrame(seasons_data)
        return df_seasons

# Example usage
if __name__ == '__main__':
    


    # Example batch (kept close to your original)
    jobs = [
        ("Netherlands", "Eredivisie"),
        ("Portugal", "Primeira Liga"),
        ("Mexico", "Liga MX"),
    ]
    jobs = [
        ("Germany", "Bundesliga"),
        ("Spain", "La Liga"),
        ("England", "Premier League"),
    ]
    jobs = [
        # already on your list
        ("Netherlands", "Eredivisie"),
        ("Portugal", "Primeira Liga"),
        ("Mexico", "Liga MX"),
    
        # strong feeders (top tiers)
        ("Belgium", "Jupiler Pro League"),        # African & South American pipeline
        ("Austria", "Bundesliga"),                # RB Salzburg, German loans
        ("Switzerland", "Super League"),          # stepping-stone for DACH/Balkans
        ("Denmark", "Superliga"),                 # exports to Bundesliga & PL
        ("Norway", "Eliteserien"),                # Ødegaard path, Premier League attention
        ("Sweden", "Allsvenskan"),                # many Bundesliga/Netherlands links
    
        # important second leagues
        ("England", "Championship"),              # extremely competitive, many loans from PL
        ("Spain", "Segunda División"),            # La Liga clubs loan out prospects here
        ("Germany", "2. Bundesliga"),             # crucial for Bundesliga player development
        ("Italy", "Serie B"),                     # used for Serie A loanees and reclamations
        ("France", "Ligue 2"),                    # big role in feeding Ligue 1 & exporting abroad
    ]

    for year in range(2023, 2020, -1):
        for country, league in jobs:
            fetcher = FootballDataFetcher(API_KEY, league, country, year)
            fetcher.run_fetch()
            time.sleep(10)  # polite delay between league requests
            
            
    # API_KEY = "427b1bc85aa3a6a81fc63b43df0dbd55"
    # # Example: Fetch data for multiple seasons (new instance per season)
    # for year in range(2024, 2013, -1):
    #     fetcher = FootballDataFetcher(API_KEY, "Bundesliga", "Germany", year)
    #     fetcher.run_fetch()
    #     time.sleep(10)  # Rate limiting delay
        
    #     fetcher = FootballDataFetcher(API_KEY, "La Liga", "Spain", year)
    #     fetcher.run_fetch()
    #     time.sleep(10)  # Rate limiting delay
        
    #     fetcher = FootballDataFetcher(API_KEY, "Premier League", "England", year)
    #     fetcher.run_fetch()
    #     time.sleep(10)  # Rate limiting delay