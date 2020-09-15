import so_study.models as models

class CivicDutyExperiment:
    model_to_test = {
        'baseline_flow_count': models.NormalizingFlowBaseline,
        'fp_flow_count': models.NormalizingFlowFP,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'data_path': "../data",
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }


class ElectorateExperiment:
    model_to_test = {
        # 'baseline_count': models.BaselineVAECount,
        # 'linear_count': models.LinearParametricVAECount,
        # 'personalised_linear_count': models.LinearParametricPlusSteerParamVAECount,
        # 'full_parameterised_count': models.FullParameterisedVAECount,
        # 'full_personalised_parameterised_count': models.FullParameterisedPlusSteerParamVAECount,
        'baseline_flow_count': models.NormalizingFlowBaseline,
        'fp_flow_count': models.NormalizingFlowFP,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'data_path': "../data",
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }


class StrunkWhiteExperiment:
    model_to_test = {
        'baseline_flow_count': models.NormalizingFlowBaseline,
        'fp_flow_count': models.NormalizingFlowFP,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'data_path': "../data",
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }


class CopyEditorExperiment:
    model_to_test = {
        'baseline_flow_count': models.NormalizingFlowBaseline,
        'fp_flow_count': models.NormalizingFlowFP,
        'full_personalised_normalizing_flow': models.NormalizingFlowFP_PlusSteer
    }
    common_params = {
        'window_length': 5*7,
        'badge_focus': 'strunk_white',
        'out_dim': 0,
        'data_path': "../data",
        'badge_threshold': 80,
        'badges_to_avoid': [],
        'ACTIONS': [0]
    }
