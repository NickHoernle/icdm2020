import fire
from so_study.load_so_data import transform_raw_edit_data_to_pt_format

if __name__ == '__main__':
    fire.Fire({
        'transform_raw_edit_data_to_pt_format': transform_raw_edit_data_to_pt_format,
    })
