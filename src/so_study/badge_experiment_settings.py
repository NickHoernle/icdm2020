class CivicDutyExperiment:
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'CivicDuty',
        'out_dim': (4, 5),
        'badge_threshold': 80,
        'badges_to_avoid': [],
    }


class ElectorateExperiment:
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'Electorate',
        'out_dim': 'QuestionVotes',
        'badge_threshold': 600,
        'badges_to_avoid': ["CivicDuty"],
    }


class StrunkWhiteExperiment:
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }


class CopyEditorExperiment:
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'copy_editor',
        'out_dim': 0,
        'badge_threshold': 300,
        'badges_to_avoid': ["strunk_white"],
        'ACTIONS': [0]
    }
