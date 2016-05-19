import csv
import os
from decimal import Decimal

from pybrain_examples.nn_prediction.exceptions import InvalidDatasetException

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
NETWORKS_DIR = os.path.join(os.path.dirname(__file__), 'networks')


class ObviousModelData(object):
    def __init__(self, home_team_shots, away_team_shots, home_team_shots_on_target, away_team_shots_on_target,
                 home_team_corners, away_team_corners, home_team_red_cards, away_team_red_cards, b365_home_win_odds,
                 b365_draw_odds, b365_away_win_odds, home_team_ht_goals, away_team_ht_goals, **kwargs):

        self.home_team_shots = Decimal(home_team_shots)
        self.away_team_shots = Decimal(away_team_shots)
        self.home_team_shots_on_target = Decimal(home_team_shots_on_target)
        self.away_team_shots_on_target = Decimal(away_team_shots_on_target)
        self.home_team_corners = Decimal(home_team_corners)
        self.away_team_corners = Decimal(away_team_corners)
        self.home_team_red_cards = Decimal(home_team_red_cards)
        self.away_team_red_cards = Decimal(away_team_red_cards)
        self.b365_home_win_odds = Decimal(b365_home_win_odds)
        self.b365_draw_odds = Decimal(b365_draw_odds)
        self.b365_away_win_odds = Decimal(b365_away_win_odds)

        self.home_team_ht_goals = Decimal(home_team_ht_goals)
        self.away_team_ht_goals = Decimal(away_team_ht_goals)

    def normalize_feature(self):
        pass


class TrainingObviousModelData(ObviousModelData):
    OUTPUT_TO_BINARY = {
        'H': 0,
        'D': 1,
        'A': 2,
    }

    def __init__(self, home_team_shots, away_team_shots, home_team_shots_on_target, away_team_shots_on_target,
                 home_team_corners, away_team_corners, home_team_red_cards, away_team_red_cards, b365_home_win_odds,
                 b365_draw_odds, b365_away_win_odds, home_team_ht_goals, away_team_ht_goals, **kwargs):

        self.output = kwargs.pop('output')

        super(TrainingObviousModelData, self).__init__(
            home_team_shots, away_team_shots, home_team_shots_on_target, away_team_shots_on_target,
            home_team_corners, away_team_corners, home_team_red_cards, away_team_red_cards,
            b365_home_win_odds, b365_draw_odds, b365_away_win_odds, home_team_ht_goals, away_team_ht_goals, **kwargs)

    @property
    def binarized_output(self):
        return self.OUTPUT_TO_BINARY.get(self.output)

    def to_list(self):
        return [value for attr, value in self.__dict__.iteritems() if attr not in ('output',)]

    @property
    def min_value(self):
        return min(self.to_list())

    @property
    def max_value(self):
        return max(self.to_list())


class FootballDataCsv(object):
    OUTPUT = 6
    HALF_TIME_HOME_GOLS = 7
    HALF_TIME_AWAY_GOLS = 8
    HOME_TEAM_SHOTS = 11
    AWAY_TEAM_SHOTS = 12
    HOME_TEAM_SHOTS_ON_TARGET = 13
    AWAY_TEAM_SHOTS_ON_TARGET = 14
    HOME_TEAM_CORNERS = 16
    AWAY_TEAM_CONVERS = 17
    HOME_TEAM_RED_CARDS = 21
    AWAY_TEAM_RED_CARDS = 22
    B365H = 23
    B365D = 24
    B365A = 25

    def __init__(self, filename):
        self.filename = filename
        self.total_min_values = []
        self.total_max_values = []
        self.data = list(self.get_game_data(filename))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def total_min(self):
        return min(self.total_min_values)

    def total_max(self):
        return max(self.total_max_values)

    def get_game_data(self, filename):
        with open(os.path.join(DATASET_DIR, filename)) as data:
            historical_data = csv.reader(data, quoting=csv.QUOTE_MINIMAL)
            if not self.is_valid(historical_data):
                raise InvalidDatasetException

            for row in historical_data:
                output = row[self.OUTPUT]
                home_team_shots = row[self.HOME_TEAM_SHOTS]
                away_team_shots = row[self.AWAY_TEAM_SHOTS]
                home_team_shots_on_target = row[self.HOME_TEAM_SHOTS_ON_TARGET]
                away_team_shots_on_target = row[self.AWAY_TEAM_SHOTS_ON_TARGET]
                home_team_corners = row[self.HOME_TEAM_CORNERS]
                away_team_corners = row[self.AWAY_TEAM_CONVERS]
                home_team_red_cards = row[self.HOME_TEAM_RED_CARDS]
                away_team_red_cards = row[self.AWAY_TEAM_RED_CARDS]
                b365_home_win_odds = row[self.B365H]
                b365_draw_odds = row[self.B365D]
                b365_away_win_odds = row[self.B365A]

                home_team_ht_goals = row[self.HALF_TIME_HOME_GOLS]
                away_team_ht_goals = row[self.HALF_TIME_AWAY_GOLS]

                model_data = TrainingObviousModelData(
                    home_team_shots, away_team_shots, home_team_shots_on_target, away_team_shots_on_target,
                    home_team_corners, away_team_corners, home_team_red_cards, away_team_red_cards,
                    b365_home_win_odds, b365_draw_odds, b365_away_win_odds, home_team_ht_goals,
                    away_team_ht_goals, output=output)
                self.total_min_values.append(model_data.min_value)
                self.total_max_values.append(model_data.max_value)
                yield model_data

    def is_valid(self, historical_data):
            header_row = next(historical_data)

            mandatory_fields = {
                'HS': FootballDataCsv.HOME_TEAM_SHOTS,
                'AS': FootballDataCsv.AWAY_TEAM_SHOTS,
            }
            for field, column in mandatory_fields.iteritems():
                if field not in header_row[column]:
                    return False
            return True
