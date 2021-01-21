import fire
from so_study.load_so_data import transform_raw_edit_data_to_pt_format
from reputation_study.data_preprosessor import create_reputation_dataset

if __name__ == '__main__':
    fire.Fire({
        'transform_raw_edit_data_to_pt_format': transform_raw_edit_data_to_pt_format,
        "create_reputation_dataset": create_reputation_dataset
    })
